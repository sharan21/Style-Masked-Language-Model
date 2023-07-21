import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from torch.autograd import Variable
from torch.nn import LSTMCell
from attention import TanhAttention
from noise import noisy, embd_noise
from sklearn.neighbors import NearestNeighbors
from nltk.translate.bleu_score import sentence_bleu 
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from vocab import Vocab

def reparameterize(mu, logvar):
	std = torch.exp(0.5*logvar)
	eps = torch.randn_like(std)
	return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
	var = torch.exp(logvar)
	logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
	return logp.sum(dim=1)

def loss_kl(mu, logvar):
	return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

def loss_bleu(refs, hyps):
	f = nltk.translate.bleu_score.SmoothingFunction(0.000000001).method1
	r = sentence_bleu([refs], hyps, smoothing_function=f)
	return(r/len(base))

def loss_ce(yhat, targets):
	criterion = nn.BCELoss(reduction='none')
	targets = targets.squeeze()
	yhat = yhat.squeeze()
	loss = criterion(yhat, targets)
	return loss.sum(dim=0)


def make_pos(t):
		return torch.sqrt(t*t)


class TextModel(nn.Module):
	"""Container module with word embedding and projection layers"""

	def __init__(self, vocab, args, initrange=0.1):
		super().__init__()
		self.vocab = vocab
		self.args = args    
		if(args.glove):
			weights = torch.FloatTensor(get_glove(vocab))            
			self.embed = nn.Embedding.from_pretrained(weights)
		else:
			self.embed = nn.Embedding(vocab.size, args.dim_emb)
		self.proj = nn.Linear(args.dim_h, vocab.size)
		self.embed.weight.data.uniform_(-initrange, initrange)
		self.proj.bias.data.zero_()
		self.proj.weight.data.uniform_(-initrange, initrange)

class DAE(TextModel):
	"""Denoising Auto-Encoder/ Parent Class of VAE/AAE/VanillaAE"""

	def __init__(self, vocab, args):
		super().__init__(vocab, args)
		self.drop = nn.Dropout(args.dropout) 
		self.E = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
			dropout=args.dropout if args.nlayers > 1 else 0, bidirectional=True)
		self.G = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
			dropout=args.dropout if args.nlayers > 1 else 0)
		self.h2mu = nn.Linear(args.dim_h*2, args.dim_z)
		self.h2logvar = nn.Linear(args.dim_h*2, args.dim_z)
		self.z2emb = nn.Linear(args.dim_z, args.dim_emb)
		self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))
		self.step_count = 0
		self.ann_weight = 0.0
		self._zeta = 0.0

	def flatten(self):
		self.E.flatten_parameters()
		self.G.flatten_parameters()

	def encode(self, input, args, is_train=False):
		embds = self.embed(input) #LBE
		if(args.zeta != 0.0 and is_train):
			self.ann_weight = get_anneal_weight(self.step_count/args.log_interval, args) if args.fn is not None else 1.0
			self._zeta = args.zeta * self.ann_weight
			noise = embd_noise(embds, args, zeta=self._zeta)
			embds  = embds + noise		
		input = self.drop(embds)
		_, (h, _) = self.E(input)
		h = torch.cat([h[-2], h[-1]], 1)
		return self.h2mu(h), self.h2logvar(h), h

	def decode(self, z, input, hidden=None):
		input = self.drop(self.embed(input)) + self.z2emb(z)
		output, hidden = self.G(input, hidden)
		output = self.drop(output)
		logits = self.proj(output.view(-1, output.size(-1)))
		return logits.view(output.size(0), output.size(1), -1), hidden

	def prof_force(self, base, outs):#base -> B, outs->B		
		base = base.to(torch.int)
		mask = torch.zeros_like(base)
		mask[base==4] = 1 #check if base token is blank
		mask_c = torch.logical_not(mask)
		outs = mask*outs + mask_c*base #only retain tokens which sit on corr. <blank> token in base
		outs = outs.to(torch.int)
		return outs.unsqueeze(dim=0)
	
	def pad(self, base):
		max_len = max([len(b) for b in base])
		n = []
		print(self.vocab.get_sent(base))
		for b in base:
			p = np.array([self.vocab.pad]*(max_len-len(b)))
			n.append(np.concatenate((b, p)))
		return np.array(n)

	def fill_in_the_blanks(self, z, base): #base -> ground truth inputs BL, z -> BZ 
		assert(len(z) == len(base))
		base = self.pad(base)
		max_len = base.shape[-1]
		base = torch.tensor(base, device=z.device)
		sents = []
		input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go) #B
		hidden = None
		for l in range(max_len): #L times
			# print(len(input))
			input = self.prof_force(base[:,l], input[0])
			# print(input.shape)
			sents.append(input) #append outs of prev generation
			logits, hidden = self.decode(z, input, hidden) #logits -> BV
			input = logits.argmax(dim=-1) #B
			
		return torch.cat(sents) #BL
	
	def generate(self, z, max_len, alg): #base -> BL
		assert alg in ['greedy' , 'sample' , 'top5']
		sents = []
		input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go) #B
		hidden = None
		for l in range(max_len): #L times
			#filter out generated tokens which are 'non-blanks'
			sents.append(input)
			logits, hidden = self.decode(z, input, hidden) #logits -> BV
			if alg == 'greedy':
				input = logits.argmax(dim=-1) #B
			elif alg == 'sample':
				input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
			elif alg == 'top5':
				not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
				logits_exp=logits.exp()
				logits_exp[:,:,not_top5_indices]=0.
				input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()
		return torch.cat(sents) #BL

	def forward(self, input, args, is_train=False):
		_input = noisy(self.vocab, input, *self.args.token_noise) if(is_train) else input
		mu, logvar, _ = self.encode(input=_input, args=args, is_train=is_train)
		z = reparameterize(mu, logvar)
		logits, _ = self.decode(z, input)
		return mu, logvar, z, logits

	def loss_rec(self, logits, targets):
		loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
			ignore_index=self.vocab.pad, reduction='none').view(targets.size())
		return loss.sum(dim=0)

	def loss(self, losses):
		return losses['rec']

	def get_anneal_info(self):
		return {"steps": self.step_count, 
				"weight": self.ann_weight, 
				"zeta": self._zeta}

	def autoenc(self, inputs, targets, args, is_train=False):
		_, _, _, logits = self(inputs, args, is_train)
		self.step_count += 1
		return {'rec': self.loss_rec(logits, targets).mean()}
	
	def step(self, losses):
		self.opt.zero_grad()
		losses['loss'].backward()
		# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
		#nn.utils.clip_grad_norm_(self.parameters(), clip)
		self.opt.step()

	def nll_is(self, inputs, targets, args, m):
		"""compute negative log-likelihood by importance sampling:
		   p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
		"""
		mu, logvar, _ = self.encode(inputs, args)
		tmp = []
		for _ in range(m):
			z = reparameterize(mu, logvar)
			logits, _ = self.decode(z, inputs)
			v = log_prob(z, torch.zeros_like(z), torch.zeros_like(z)) - \
				self.loss_rec(logits, targets) - log_prob(z, mu, logvar)
			tmp.append(v.unsqueeze(-1))
		ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
		return -ll_is

class VAE(DAE):
	"""Variational Auto-Encoder"""

	def __init__(self, vocab, args):
		assert args.lambda_kl != 0.0
		super().__init__(vocab, args)

	def loss(self, losses):
		return losses['rec'] + self.args.lambda_kl * losses['kl']

	def autoenc(self, inputs, args, targets, is_train=False):
		mu, logvar, _, logits = self(input=inputs, args=args, is_train=is_train)
		self.step_count += 1
		return {'rec': self.loss_rec(logits, targets).mean(),
				'kl': loss_kl(mu, logvar)}

class AAE(DAE):
	"""Adversarial Auto-Encoder"""

	def __init__(self, vocab, args):
		if(args.lambda_adv == 0.0):
			print("Note: You are training a AAE with lambda_adv = 0.0")
		super().__init__(vocab, args)
		self.D = nn.Sequential(nn.Linear(args.dim_z, args.dim_d), nn.ReLU(),
			nn.Linear(args.dim_d, 1), nn.Sigmoid())
		self.optD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))

	def loss_adv(self, z):
		zn = torch.randn_like(z)
		zeros = torch.zeros(len(z), 1, device=z.device)
		ones = torch.ones(len(z), 1, device=z.device)
		loss_d = F.binary_cross_entropy(self.D(z.detach()), zeros) + \
			F.binary_cross_entropy(self.D(zn), ones)
		loss_g = F.binary_cross_entropy(self.D(z), ones)
		return loss_d, loss_g

	def loss(self, losses):
		return losses['rec'] + self.args.lambda_adv * losses['adv'] + \
			self.args.lambda_p * losses['|lvar|']

	def autoenc(self, inputs, targets, lbls, args, is_train=False):
		_, logvar, z, logits = self(inputs, args, is_train=is_train)
		loss_d, adv = self.loss_adv(z)
		self.step_count += 1
		return {'rec': self.loss_rec(logits, targets).mean(),
				'adv': adv,
				'|lvar|': logvar.abs().sum(dim=1).mean(),
				'loss_d': loss_d}

	def step(self, losses):
		super().step(losses)
		self.optD.zero_grad()
		losses['loss_d'].backward()
		self.optD.step()

class VanillaAE(DAE):
	"""Deterministic Auto-Encoder"""

	def __init__(self, vocab, args):
		super().__init__(vocab, args)
		self.h2ls = nn.Linear(args.dim_h*2, args.dim_z)

	def encode(self, input, args, is_train=False):
		embds = self.embed(input) #LBE
		if(args.zeta != 0.0 and is_train):
			self.ann_weight = get_anneal_weight(self.step_count/args.log_interval, args) if args.fn is not None else 1.0
			self._zeta = args.zeta * self.ann_weight
			noise = embd_noise(embds, noise_type=args.noise_type, zeta=self._zeta)
			embds  = embds + noise
			input = self.drop(embds)	
		input = self.drop(embds)        
		_, (h, _) = self.E(input)
		h = torch.cat([h[-2], h[-1]], 1)
		return self.h2ls(h), None, h

	def forward(self, input, args, is_train=False):
		_input = noisy(self.vocab, input, *self.args.token_noise) if(is_train) else input
		z, _, _ = self.encode(_input, args, is_train=is_train)
		logits, _ = self.decode(z, input)
		return z, logits

	def loss(self, losses):
		return losses['rec']

	def autoenc(self, inputs, targets, lbls, args, is_train=False):
		z, logits = self(inputs, args, is_train)
		self.step_count += 1
		return {'rec': self.loss_rec(logits, targets).mean()}


class LSTMClassifier(TextModel):
	def __init__(self, vocab, args):
		super().__init__(vocab, args)

		self.drop = nn.Dropout(args.dropout) 
		self.output_size = args.output_size 
		self.E = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers, bidirectional=True)            
		self.h2out = nn.Linear(args.dim_h*2, self.output_size)
		self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))
	
	def flatten(self):
		self.E.flatten_parameters()

	def forward(self, input, args, is_train=False, skip_encode=False):
		input = self.drop(self.embed(input)) #LBE
		_, (h, _) = self.E(input) #2*BH
		h = torch.cat([h[-2], h[-1]], 1) #B(H*2)    
		out = self.h2out(h) #BO
		out = nn.Softmax(dim=1)(out)
		return None, None, out

	def loss(self, losses):
		return losses['ce']

	def autoenc(self, inputs, targets, lbls, args, is_train=False, skip_encode=False): # return dict of 'losses'
		_, _, outs = self(inputs, args=args, is_train=is_train, skip_encode=skip_encode)
		return {'ce': loss_ce(outs, lbls).mean()}

	def step(self, losses):
		self.opt.zero_grad()
		losses['loss'].backward()
		self.opt.step()
	
class DiversityLSTMClassifier(LSTMClassifier): #with reparametrization layer after embeddings, X(tokens)->E(embds)->Z(latent embds)->H(hidden)->Y(outs)

	def __init__(self, vocab, args):
		super().__init__(vocab, args)
		# assert args.lambda_c != 0.0
		self.drop = nn.Dropout(args.dropout) 
		self.output_size = args.output_size 
		self.E = nn.LSTM(args.dim_z, args.dim_h, args.nlayers, bidirectional=True, batch_first=True)            
		self.h2out = nn.Linear(args.dim_h*2, self.output_size)
		self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))
		self.attention = TanhAttention(args.dim_h*2)
		self.e2mu = nn.Linear(args.dim_emb, args.dim_z)
		self.e2logvar = nn.Linear(args.dim_emb, args.dim_z)
		self.D = nn.Sequential(nn.Linear(args.dim_z, args.dim_d), nn.ReLU(),
			nn.Linear(args.dim_d, 1), nn.Sigmoid())
		self.optD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))
	
	def flatten(self):
		self.E.flatten_parameters()
	
	def encode(self, input, args, is_train=False, skip_encode=False):
		if(skip_encode):
			self.z = input #input is latent embeddings BLZ
			self.z.requires_grads = True
			self.z.retain_grads = True
			return None, None, None
		else:
			embds = self.embed(input) #BLE
			logvar = self.e2logvar(embds)
			mu = self.e2mu(embds)
			self.z = reparameterize(mu, logvar) if is_train else mu
			self.z.requires_grads = True
			self.z.retain_grads = True
			# self.emb = embds #saving for use in ig experiment
			return mu, logvar, None

	def forward(self, input, args, is_train=False, skip_encode=False):
		if(not skip_encode):
			input = input.t() #BL, we are using batch_first=True for LSTM
		mu, logvar, _ = self.encode(input, args, is_train=is_train, skip_encode=skip_encode)
		if(is_train):
			self.z = self.drop(self.z) #BLZ
		self.hs, (h, _) = self.E(self.z) #h -> 2*BH
		h = torch.cat([h[-2], h[-1]], 1) #final hidden state B(H*2), save this to compute conicity later  
		attn = self.attention(self.hs).unsqueeze(dim=-1) #hs->BLH, attn->BL1
		hidden = attn * self.hs #BLH	
		hidden = torch.sum(hidden, dim=1) #BH
		out = self.h2out(hidden) #BO
		out = nn.Softmax(dim=1)(out)
		self.attn = attn.detach().squeeze() #save this if needed later
		return mu, logvar, out

	
	def loss_adv(self, z):
		zn = torch.randn_like(z)
		zeros = torch.zeros(z.shape[0], z.shape[1], 1, device=z.device)
		ones = torch.ones(z.shape[0], z.shape[1], 1, device=z.device)
		loss_d = F.binary_cross_entropy(self.D(z.detach()), zeros) 
		loss_d += F.binary_cross_entropy(self.D(zn), ones)
		loss_g = F.binary_cross_entropy(self.D(z), ones)
		return loss_d, loss_g

	def compute_conicity(self):#find conicity of the batch LBH
		self.hs = torch.permute(self.hs, (1,0,2)) #BLH
		mean = torch.mean(self.hs, dim=1).unsqueeze(dim=1) #B1H
		atm = nn.CosineSimilarity(dim=-1)(self.hs, mean).squeeze() #BL
		mean_atm = atm.mean(dim=-1) #B
		con = mean_atm.mean() #1	
		return con

	def make_pos(self, t):
		return torch.sqrt(t*t)

	def loss(self, losses):
		return losses['ce'] + self.args.lambda_c * losses['conicity'] + self.args.lambda_adv * losses['adv']

	def autoenc(self, inputs, targets, lbls, args, is_train=False, skip_encode=False): # return dict of 'losses'
		_, logvar, outs = self(inputs, args=args, is_train=is_train, skip_encode=skip_encode)
		loss_d, adv = self.loss_adv(self.z)
		return {'ce': loss_ce(outs, lbls).mean(),
				'conicity':self.make_pos(self.compute_conicity()), #to prevent conicity -> -ve
				'adv': adv,
				'|lvar|': logvar.abs().sum(dim=1).mean() if logvar is not None else None,
				'loss_d': loss_d
				}

	def step(self, losses):
		super().step(losses) #step for losses['loss']
		self.optD.zero_grad()
		losses['loss_d'].backward()
		self.optD.step()

############################################################################## transformer models
class PositionalEncoding(nn.Module):
	r"""Inject some information about the relative or absolute position of the tokens
		in the sequence. The positional encodings have the same dimension as
		the embeddings, so that the two can be summed. Here, we use sine and cosine
		functions of different frequencies.
	.. math::
		\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
		\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
		\text{where pos is the word position and i is the embed idx)
	Args:
		d_model: the embed dim (required).
		dropout: the dropout value (default=0.1).
		max_len: the max. length of the incoming sequence (default=5000).
	Examples:
		>>> pos_encoder = PositionalEncoding(d_model)
	"""

	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		r"""Inputs of forward function
		Args:
			x: the sequence fed to the positional encoder model (required).
		Shape:
			x: [sequence length, batch size, embed dim]
			output: [sequence length, batch size, embed dim]
		Examples:
			>>> output = pos_encoder(x)
		"""

		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

class TransformerModel(TextModel):
	"""Container module with an encoder, a recurrent or transformer module, and a decoder."""

	def __init__(self, vocab, args):
		super(TransformerModel, self).__init__(vocab, args)
		self.src_mask = None
		self.pos_encoder = PositionalEncoding(args.dim_emb, args.dropout)
		encoder_layers = TransformerEncoderLayer(args.dim_emb, args.nhead, args.dim_ff, args.dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, args.nlayers)
		if(False):
			self.decoder = nn.Linear(args.dim_emb, vocab.size)
		self.e2v = nn.Linear(args.dim_emb, vocab.size)
		self.opt = optim.Adam([
				{'params': self.transformer_encoder.parameters(), 'lr': args.lr},
				{'params': self.e2v.parameters(), 'lr': args.lr}
			], lr=args.lr, betas=(0.5, 0.999))
		#adv style classifier params
		self.style_dis = nn.Sequential(nn.Linear(args.dim_emb, args.dim_d), nn.Linear(args.dim_d, args.output_size), nn.Softmax(dim=-1))
		self.opt_style_dis = optim.Adam(self.style_dis.parameters(), lr=args.lr, betas=(0.5, 0.999))
		#adv fluency classifier params
		self.fluency_dis = nn.Sequential(nn.Linear(args.dim_emb, args.dim_d), nn.Linear(args.dim_d, args.output_size), nn.Softmax(dim=-1))
		self.opt_fluency_dis = optim.Adam(self.fluency_dis.parameters(), lr=args.lr, betas=(0.5, 0.999))
		
	def generate_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask
	
	def print_params(self):
		for name, param in self.named_parameters():
			if param.requires_grad:
				print(name)

	def compute_acc(self, yhat, y):
		n_samples = y.size(0)
		yhat = torch.argmax(yhat, dim=-1)
		y = torch.argmax(y, dim=-1)    
		return torch.tensor(yhat[yhat == y].size(0)/n_samples)
	
	def loss_rec(self, logits, targets):
		loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
			ignore_index=self.vocab.pad, reduction='none').view(targets.size())
		return loss.sum(dim=0)
	
	def concat_meta(self, inputs, targets, lbls):#inputs, targets -> LB; lbls -> BO
		dst_lbls = lbls.argmax(dim=-1).detach().to(torch.long) + self.vocab.nmeta #B
		dst_lbls = dst_lbls.unsqueeze(dim=0) #1B
		cls_token = torch.zeros(1, len(lbls), device=lbls.device).to(torch.long) + self.vocab.s_cls #1B
		inputs, targets = torch.cat((cls_token, inputs, dst_lbls), dim=0), torch.cat((cls_token, targets, dst_lbls), dim=0)		
		return inputs, targets

	def loss_fluency(self, embd_inv):
		b, e = embd_inv.shape
		ones = torch.ones(b, 2, device=embd_inv.device) #1->src 
		return F.binary_cross_entropy(self.fluency_dis(embd_inv), ones) #fool fluency disc
	
	def loss_fluency_dis(self, embd_rec, embd_inv): #embd -> BE
		b, e = embd_rec.shape
		zeros = F.one_hot(torch.zeros(b, dtype=torch.long, device=embd_rec.device), num_classes=2).to(torch.float)
		ones = F.one_hot(torch.ones(b, dtype=torch.long, device=embd_rec.device), num_classes=2).to(torch.float)
		return F.binary_cross_entropy(self.fluency_dis(embd_rec), ones) + 0.1 * F.binary_cross_entropy(self.fluency_dis(embd_inv), zeros)

	def loss_sta(self, lbls, preds): 
		return F.binary_cross_entropy(preds, lbls)
		
	def loss_style_dis(self, lbls, lbls_inv, preds_rec, preds_inv): 
		return F.binary_cross_entropy(preds_rec, lbls) - 0.1 * F.binary_cross_entropy(preds_inv, lbls_inv)
	
	def add_blanks(self, inputs, targets):
		l, b = inputs.shape
		n_pads = int(self.args.lambda_pad * l)
		index = random.sample(range(0, l-1), n_pads)
		pad = torch.full((1, b), self.vocab.word2idx['<blank>']).to(inputs.device)
		def pad_vector(x):
			for idx in index:
				x = torch.cat((x[:idx],pad,x[idx:]), dim=0)
			return x
		
		inputs = pad_vector(inputs)
		t = pad.repeat(n_pads, 1)
		targets = torch.cat((targets, t), dim=0)
		assert inputs.shape == targets.shape
		return inputs, targets
			
	def autoenc(self, inputs, targets, lbls, args, is_train=False): # return dict of 'losses'
		if(args.lambda_pad > 0 and not is_train):
			inputs, targets = self.add_blanks(inputs, targets)
		if(args.fine_tune):
			# assert self.args.lambda_sta != 0.0 and self.args.lambda_fluency != 0.0 and self.args.lambda_rec_inv != 0.0
			self.args.lambda_rec = 0.0
		else:
			assert self.args.lambda_sta == 0.0 and self.args.lambda_fluency == 0.0 and self.args.lambda_rec_inv == 0.0
		#reconstruct 
		inputs_rec, targets_rec = self.concat_meta(inputs, targets, lbls) #concat dst lbls to inputs
		embds_rec, self.logits_rec = self(inputs_rec, args=args, is_train=is_train) #get reconstructed tokens
		#style invert
		lbls_inv = torch.logical_not(lbls).to(torch.float)	# only works for 2 lbls
		inputs_inv, targets_inv = self.concat_meta(inputs, targets, lbls_inv) #concat dst lbls to inputs
		embds_inv, self.logits_inv = self(inputs_inv, args=args, is_train=is_train) #get reconstructed tokens
		
		return {'rec': self.loss_rec(self.logits_rec, targets_rec).mean(),
				'rec_inv': self.loss_rec(self.logits_inv, targets_inv).mean(),
				'loss_sta': self.loss_sta(lbls_inv, self.style_dis(embds_inv[:-2].mean(dim=0))).mean(), #style
				'loss_style_dis': self.loss_style_dis(lbls, lbls_inv,  self.style_dis(embds_rec[:-2].mean(dim=0).detach()), self.style_dis(embds_inv[:-2].mean(dim=0).detach())).mean(),
				'style_acc_rec': self.compute_acc(self.style_dis(embds_rec[:-2].mean(dim=0)), lbls), 
				'style_acc_inv': self.compute_acc(self.style_dis(embds_inv[:-2].mean(dim=0)), lbls_inv), 
				# 'loss_fluency': self.loss_fluency(embds_inv[:-2].mean(dim=0)).mean(), #fluency
				# 'loss_fluency_dis': self.loss_fluency_dis(embds_rec[:-2].mean(dim=0).detach(), embds_inv[:-2].mean(dim=0).detach()).mean(),
				# 'fluency_acc_rec': self.compute_acc(fluency_preds_rec, torch.ones_like(fluency_preds_rec)), 
				# 'fluency_acc_inv': self.compute_acc(fluency_preds_inv, torch.zeros_like(fluency_preds_inv)) 
				}

	def forward(self, input, args, is_train=False, has_mask=False):
		if has_mask:
			device = input.device
			if self.input_mask is None or self.input_mask.size(0) != len(input):
				mask = self.generate_mask(len(input)).to(device)
				self.input_mask = mask #BB
		else:
			self.input_mask = None
		input = self.embed(input) * math.sqrt(self.args.dim_emb) #LBE        
		input = self.pos_encoder(input)
		embds = self.transformer_encoder(input, self.input_mask) #LBE
		if(False): #generational
			outs = self.decoder(embds) #LBV
			outs = F.log_softmax(outs, dim=-1)
		else: #only encoder
			outs = F.log_softmax(self.e2v(embds), dim=-1) #LBV
		return embds, outs

	def loss(self, losses):
		return self.args.lambda_rec * losses['rec'] +\
			self.args.lambda_rec_inv * losses['rec_inv']+ \
			self.args.lambda_sta * losses['loss_sta']
			# self.args.lambda_fluency * losses['loss_fluency']
	
	def step(self, losses):
		#step for transformer encoder
		self.opt.zero_grad()
		losses['loss'].backward()
		if(self.args.fine_tune):
			torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
		self.opt.step()
		#step for style dis
		self.opt_style_dis.zero_grad()
		losses['loss_style_dis'].backward()
		self.opt_style_dis.step()
		#step for fluency dis
		# self.opt_fluency_dis.zero_grad()
		# losses['loss_fluency_dis'].backward()
		# self.opt_fluency_dis.step()