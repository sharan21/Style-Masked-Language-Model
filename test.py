import argparse
import os
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from nltk.translate.bleu_score import sentence_bleu 
from vocab import Vocab
from model import *
from utils import *
from preproc import create_toy_dataset
from batchify import *
from nlgeval import compute_metrics
import nltk

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', metavar='DIR', default=None,
					help='Name of dataset for bert pretrained')
parser.add_argument('--checkpoint', metavar='DIR', default=None,
					help='checkpoint directory')
parser.add_argument('--classifier-checkpoint', metavar='DIR', default=None,
					help='checkpoint directory')
parser.add_argument('--output', metavar='FILE', default='output',
					help='output file name (in checkpoint directory)')
parser.add_argument('--data', metavar='FILE',
					help='path to data file')
parser.add_argument('--enc', default='mu', metavar='M',
					choices=['mu', 'z'],
					help='encode to mean of q(z|x) or sample z from q(z|x)')
parser.add_argument('--dec', default='greedy', metavar='M',
					choices=['greedy', 'sample'],
					help='decoding algorithm')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
					help='batch size')
parser.add_argument('--max-len', type=int, default=35, metavar='N',
					help='max sequence length')
parser.add_argument('--dataset', default=None, metavar='M')
parser.add_argument('--evaluate', action='store_true',
					help='evaluate on data file')
parser.add_argument('--attribution', action='store_true',
					help='Generate attribution scores for a task')
parser.add_argument('--gradientascent', action='store_true',
					help='evaluate on data file')
parser.add_argument('--ppl', action='store_true',
					help='compute ppl by importance sampling')
parser.add_argument('--bert-ppl', action='store_true',
					help='compute ppl by importance sampling using pretrained bert')
parser.add_argument('--reconstruct', action='store_true',
					help='reconstruct data file')
parser.add_argument('--cond-reconstruct-ae', action='store_true',
					help='cond reconstruct on style masked dataset')
parser.add_argument('--cond-reconstruct-tr', action='store_true',
					help='cond reconstruct on style masked dataset using transformer model')
parser.add_argument('--cond-reconstruct-tr-single', action='store_true',
					help='cond reconstruct on style masked dataset using transformer model')
parser.add_argument('--bleu', action='store_true',
					help='compute bleu between source and converted sentences')
parser.add_argument('--content-cos-sim', action='store_true',
					help='compute cos sim between source and converted sent embds')
parser.add_argument('--hs-cos-sim', action='store_true',
					help='compute cos sim between source and converted hidden states')
parser.add_argument('--sample', action='store_true',
					help='sample sentences from prior')
parser.add_argument('--tsne', action='store_true',
					help='plot tsne of lspace wrt labels')
parser.add_argument('--tst-on-test', action='store_true',
					help='TST exp for a particular dataset')
parser.add_argument('--ig', action='store_true',
					help='perform style-masking exp with integrated gradients')
parser.add_argument('--grads', action='store_true',
					help='perform style-masking exp with vanilla gradients')
parser.add_argument('--mul-x', action='store_true',
					help='perform style-masking exp with vanilla gradients * X')
parser.add_argument('--mt-tst-on-test-ae', action='store_true',
					help='TST exp for a particular dataset (style-masked)')
parser.add_argument('--mt-tst-on-test-tr', action='store_true',
					help='TST exp for a particular dataset (style-masked) using transformers')
parser.add_argument('--mt-tst-on-test-tr-single', action='store_true',
					help='TST exp for a particular dataset (style-masked) using transformers')
parser.add_argument('--toy-tsne', action='store_true',
					help='plot tsne of lspace wrt labels for toy dataset')
parser.add_argument('--arithmetic', action='store_true',
					help='compute vector offset avg(b)-avg(a) and apply to c')
parser.add_argument('--complexity', action='store_true',
					help='the mean euclidean distance btw 1000 samples of both classes')
parser.add_argument('--classify', action='store_true',
					help='compute vector offset avg(b)-avg(a) and apply to c')
parser.add_argument('--bert-classify', action='store_true',
					help='compute vector offset avg(b)-avg(a) and apply to c')
parser.add_argument('--bert-pretrained', action='store_true', default=None)
parser.add_argument('--interpolate', action='store_true',
					help='interpolate between pairs of sentences')
parser.add_argument('--latent-nn', action='store_true',
					help='find nearest neighbor of sentences in the latent space')
parser.add_argument('--ppl-on-test', action='store_true',
					help='find nearest neighbor of sentences in the latent space')
parser.add_argument('--m', type=int, default=100, metavar='N',
					help='num of samples for importance sampling estimate')
parser.add_argument('--n', type=int, default=5, metavar='N',
					help='num of sentences to generate for sample/interpolate')
parser.add_argument('--k', type=float, default=2, metavar='R',
					help='k * offset for vector arithmetic')
parser.add_argument('--seed', type=int, default=1,
					metavar='N', help='random seed')
parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')
parser.add_argument('--embedding-tsne', action='store_true', help='plot tsne of all embeddings')
parser.add_argument('--clean-dataset', action='store_true', help='disable CUDA')
parser.add_argument('--output-size', type=int, default=2,
					metavar='N', help='number of classes in dataset')
parser.add_argument('--glove', '-g', default=False, action='store_true')
parser.add_argument('--nlg-eval-on-test', action='store_true',
					help='find nearest neighbor of sentences in the latent space')
parser.add_argument('--mt-nlg-eval-on-test-tr-single', action='store_true',
					help='find nearest neighbor of sentences in the latent space')
parser.add_argument('--mt-nlg-eval-on-test', action='store_true',
					help='nlg evals for style masking dataset')
parser.add_argument('--ppl-on-test-gpt', action='store_true',
					help='find nearest neighbor of sentences in the latent space')
parser.add_argument('--gpu', type=int, default=0,
					metavar='N', help='ID of gpu')
parser.add_argument('--eps', type=float, default=0.0,
					metavar='N', help='used in style masking')
parser.add_argument('--model-name', metavar='FILE',
					default='model_best.pt', help='name of model')
parser.add_argument('--kmeans', action='store_true',
					help='perform kmeans on latent space')
parser.add_argument('--style-masking', action='store_true',
					help='produce a style masked version of a dataset')
parser.add_argument('--create-forms', action='store_true', help='create forms')


def get_model(path, vocab):
	print("loading model from path {}".format(path))
	ckpt = torch.load(path)
	train_args = ckpt['args']    
	model = {'dae': DAE, 'vae': VAE, 'aae': AAE, 'van': VanillaAE, 
		'lstmclassifier': LSTMClassifier, 'divlstmclassifier':DiversityLSTMClassifier, 
		'tr-encoder':TransformerModel}[ 
		train_args.model_type](vocab, train_args).to(device)
	model.load_state_dict(ckpt['model'])
	if(not isinstance(model, TransformerModel)):
		model.flatten()
	if(not (args.gradientascent or args.style_masking)):
		model.eval()
	return model, train_args


def encode(sents):
	assert args.enc == 'mu' or args.enc == 'z'
	batches, order = get_batches(sents, vocab, args.batch_size, device)
	z = []
	for inputs, _ in batches:

		mu, logvar, _ = model.encode(inputs, train_args, is_train=False)
		if args.enc == 'mu':
			zi = mu
		else:
			zi = reparameterize(mu, logvar)
		z.append(zi.detach().cpu().numpy())
	z = np.concatenate(z, axis=0)
	z_ = np.zeros_like(z)
	z_[np.array(order)] = z
	return z_


def encode_annotated(sents):
	assert args.enc == 'mu' or args.enc == 'z'
	batches, order = get_batches_annotated(
		sents, vocab, args.batch_size, device)
	z, l = [], []
	for inputs, targets, _ in batches:
		mu, logvar, _ = model.encode(inputs, train_args, is_train=True)
		zi = mu if args.enc == 'mu' else reparameterize(mu, logvar)
		z.append(zi.detach().cpu().numpy())
		l.append(targets.detach().cpu().numpy())
	z, l = np.concatenate(z, axis=0), np.concatenate(l, axis=0)
	z_, l_ = np.zeros_like(z), np.zeros_like(l)
	z_[np.array(order)] = z
	l_[np.array(order)] = l
	return z_, l_

def encode_mt(data):
	assert args.enc == 'mu' or args.enc == 'z'
	batches, order = get_batches_mt(data, vocab, args.batch_size, device, sort=False)
	z, base = [], []
	for inputs, targets, lbls in batches:
		mu, logvar, _ = model.encode(inputs, train_args, is_train=False)
		zi = mu if args.enc == 'mu' else reparameterize(mu, logvar) #BZ
		z.append(zi.detach().cpu().numpy())
		base.extend(inputs.t().detach().cpu().numpy())
	z = np.concatenate(z, axis=0) #NZ
	z_= np.zeros_like(z)
	z_[np.array(order)] = z
	base = np.array(base)
	base[np.array(order)] = base #NL
	return z_, base

def classify(sents, model):
	assert(args.classifier_checkpoint != None)
	print("Predicting labels for {} sentences...".format(len(sents)))
	batches, _ = get_batches(sents, classifier_vocab,args.batch_size, device,sort=False)
	preds = []
	for i, (inputs, _) in enumerate(batches):
		_, _, outs = model(inputs, args)  # BO
		outs = torch.argmax(outs, dim=-1)  # B1
		preds.append(outs.detach().cpu().numpy())
	preds = np.concatenate(preds, axis=0)
	return preds

def integrated_gradients(model, inputs, targets, lbls, train_args, steps=5): #perform ig for a batch of sentences and return attr
	base = torch.zeros_like(model.z)
	final = model.z.detach()
	base.requires_grad = True
	delta = (final-base)/steps
	path_grads = torch.zeros_like(final)
	for step in range(steps):
		losses = model.autoenc(inputs=base, args=train_args, targets=targets, lbls=lbls, is_train=True, skip_encode=True)
		losses['loss'] = model.loss(losses)
		losses['loss'].backward(create_graph=True)
		dz = torch.autograd.grad(losses['loss'], base)[0] #BLZ
		path_grads += dz
		model.zero_grad()
		base = base.detach() #start fresh with new graph
		base += delta
	attr = (path_grads/steps).sum(dim=-1)
	attr = nn.Softmax(dim=-1)(attr)
	return attr

def vanilla_gradients(model, losses): #perform ig for a batch of sentences and return attr
	losses['loss'] = model.loss(losses)
	losses['loss'].backward(create_graph=True)
	dz = torch.autograd.grad(losses['loss'], model.z)[0] #BLZ
	if(args.mul_x): #dz * z
		dz = dz * model.z
	attr = nn.Softmax(dim=-1)(dz.sum(dim=-1))
	return attr

def style_masking(sents, model): #list of sents -> (inputs, targets, lbls)
	batches, order = get_batches_annotated(sents, vocab, args.batch_size, device, num_classes=args.output_size)
	inputs_c, targets_c, lbls_c = [], [], []
	for i, (inputs, targets, lbls) in tqdm(enumerate(batches), total=len(batches)):
		losses = model.autoenc(inputs=inputs, args=train_args, targets=targets, lbls=lbls, is_train=True)
		if(args.ig): #calc integrated gradients
			model.attn = integrated_gradients(model, inputs, targets, lbls, train_args)
		elif(args.grads):
			model.attn = vanilla_gradients(model, losses)
		#else use explainable attn
		mask = get_masks_from_attn_batch(model.attn, eps=args.eps) #BL
		z = torch.zeros_like(mask) #BL
		inputs = inputs.t() #BL
		targets_c.extend(inputs.cpu().numpy()[:,1:]) #splicing out <go>
		inputs[mask > z] = vocab.blank #impose masks
		inputs_c.extend(inputs.cpu().numpy()[:,1:])
		lbls_c.extend(torch.argmax(lbls, dim=-1).cpu().numpy())
	inputs_c, targets_c, lbls_c = np.array(inputs_c, dtype=object), np.array(targets_c, dtype=object), np.array(lbls_c, dtype=object), 
	inputs_c_2, targets_c_2, lbls_c_2 = inputs_c.copy(), targets_c.copy(), lbls_c.copy()
	inputs_c_2[np.array(order)], targets_c_2[np.array(order)], lbls_c_2[np.array(order)] = inputs_c, targets_c, lbls_c
	assert len(inputs_c_2) == len(targets_c_2) == len(lbls_c_2) 
	return inputs_c_2, targets_c_2, lbls_c_2

def gradientascent(sents, steps=5, lr=1): #perform TST using ga sentence by sentence
	args.batch_size = 1
	token_lookup = NearestTokenLookUp(model).to(device) 
	test_batches, _ = get_batches_annotated(sents, vocab, args.batch_size, device, num_classes=args.output_size, sort=False, no_stokens=True)
	base, base_lbls, conv, conv_lbls, attn, masks = [],[],[],[],[], []
	count = 0
	for inputs, targets, lbls in tqdm(test_batches, total=int(args.m/args.batch_size)-1):
		lbls = torch.unsqueeze(lbls, dim=0) #because we are using batchsize = 1
		#perform first step of gradient ascent on this batch
		curr_step = 0
		dest_lbls = 1 - lbls        
		losses = model.autoenc(inputs=inputs, args=train_args, targets=targets, lbls=dest_lbls, is_train=False)
		mask = get_masks_from_attn(model.attn) #L1
		attn.append(model.attn.cpu().detach().numpy()) 
		masks.append(mask.cpu().detach().numpy())
		eng_old = model.vocab.get_sent(inputs.t())
		losses['loss'] = model.loss(losses)
		losses['loss'].backward(create_graph=True)
		delta_z = torch.autograd.grad(losses['loss'], model.z)[0] #model.z stores z LZ of last forward prop
		model.z = model.z - lr * mask * delta_z
		
		# perform gradient ascent for remaining steps
		while(curr_step < steps):
			losses = model.autoenc(inputs=model.z, args=train_args, targets=targets, lbls=dest_lbls, is_train=False, skip_encode=True)
			losses['loss'] = model.loss(losses)
			losses['loss'].backward(create_graph=True)
			delta_z = torch.autograd.grad(losses['loss'], model.z)[0] #get dL/dZ
			model.z = model.z - lr * mask * delta_z
			model.zero_grad() #to prevent grads accumulating on subsequent backward calls
			curr_step += 1

		_, _, outs = model(model.z, args=train_args, is_train=True, skip_encode=True)
		z = token_lookup.get_nearest_z(model.z.cpu().detach().numpy())
		idxs = token_lookup.get_sents_from_z(z)
		eng_new = model.vocab.get_sent(idxs)
		base.extend(eng_old)
		base_lbls.extend(lbls)
		conv.extend(eng_new)
		conv_lbls.extend(outs)
		count += args.batch_size
		if(count >= args.m):
			break

	return base, base_lbls, conv, conv_lbls, attn, masks    

def decode(z):
	sents = []
	i = 0
	while i < len(z):
		zi = torch.tensor(z[i: i+args.batch_size], device=device)
		outputs = model.generate(zi, args.max_len, args.dec).t()
		for s in outputs:
			sents.append([vocab.idx2word[id] for id in s[1:]])  # skip <go>
		i += args.batch_size
	return strip_eos(sents)

def decode_mt_ae(z, base=None, cond_recon=False): #decode using ae and teacher forcing for non blank words
	sents = []
	i = 0
	while i < len(z):
		zi = torch.tensor(z[i: i+args.batch_size], device=device)
		bi = base[i:i+args.batch_size]
		outputs = model.fill_in_the_blanks(zi, bi).t() 
		for s in outputs:
			sents.append([vocab.idx2word[id] for id in s[1:] if id !=vocab.pad])  # skip <go>
		i += args.batch_size
	return strip_eos(sents)

def decode_mt_tr(data, model): #decode for transformer
	batches, order = get_batches_mt(data, model.vocab, args.batch_size, device, sort=False)
	sents_rec, base = [], []
	for inputs, targets, lbls in batches:
		outs = model(inputs, train_args, is_train=False)[1].argmax(dim=-1).t().tolist() #BL
		l = strip_eos(model.vocab.get_sent(outs))
		l = [e.split() for e in l]
		sents_rec.extend(l)
	return sents_rec

def decode_mt_tr_single(data, model): #decode for transformer, style invert
	batches, order = get_batches_mt(data, model.vocab, args.batch_size, device, sort=False)
	sents_rec, base = [], []
	for inputs, targets, lbls in batches:
		dst_lbls = torch.logical_not(lbls).to(torch.long)	# only works for 2 lbls
		inputs, _ = model.concat_meta(inputs, targets, dst_lbls)
		outs = model(inputs, train_args, is_train=False)[1].argmax(dim=-1).t().tolist() #BL
		outs = torch.tensor(outs)[:,1:] #remove <s_cls> token
		l = strip_eos(model.vocab.get_sent(outs))
		l = [e.split() for e in l]
		sents_rec.extend(l)
	return sents_rec

def calc_ppl(sents):
	batches, _ = get_batches(sents, vocab, args.batch_size, device)
	ppl = []
	tot_ppl = 0
	with torch.no_grad():
		for inputs, targets in batches:
			_, loss = model.autoenc(
				inputs=inputs, args=train_args, targets=targets)
			tot_ppl += torch.exp(loss).sum()
			ppl.extend(torch.exp(loss))

	assert(len(ppl) == len(sents))
	mean = tot_ppl/len(sents)
	res = []
	for i, sent in enumerate(sents):
		l = ' '.join(sent) + ' , ' + str(ppl[i].item()) + '\n'
		res.append(l)
	res.append("Mean ppl of {} is {}".format(len(sents), tot_ppl/len(sents)))
	return res, mean


if __name__ == '__main__':
	args = parser.parse_args()

	set_seed(args.seed)
	cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device('cuda:{}'.format(args.gpu)) if cuda else 'cpu'

	# load pretrained models, vocabs and training args
	if(args.checkpoint != None):
		vocab = Vocab(os.path.join(args.checkpoint, 'vocab.txt'))
		model, train_args = get_model(os.path.join(args.checkpoint, 'checkpoints', args.model_name), vocab)
	if(args.classifier_checkpoint != None): #used in TST exp mainly
		classifier_vocab = Vocab(os.path.join(args.classifier_checkpoint, 'vocab.txt'))
		classifier_model, classifier_train_args = get_model(os.path.join(args.classifier_checkpoint, 'checkpoints', 'model_best.pt'), classifier_vocab)
	
	# perform experiments
	if args.evaluate:
		sents = load_sent(args.data)
		batches, _ = get_batches(sents, vocab, args.batch_size, device)
		meters =    evaluate(model, batches, args)
		print(' '.join(['{} {:.2f},'.format(k, meter.avg)
						for k, meter in meters.items()]))

	if args.tst_on_test:  # TST experiment on test.csv of dataset on unsupervised dataset
		expd = os.path.join(os.getcwd(), args.checkpoint,'exps_'+args.model_name)
		print(expd)
		if not os.path.exists(expd):
			os.mkdir(expd)
		# args.checkpoint += '/experiments-'+args.model_name
		base = './data/annotated/{}/'.format(args.dataset)
		n_wrong, tot = 0, 0
		for l in range(2):  # 2 labels
			# create test.(src)to(dst)
			src_lbl = str(l)
			dst_lbl = '1' if src_lbl == '0' else '0'
			fa, fb, fc = os.path.join(base, 'test.'+src_lbl), os.path.join(
				base, 'test.'+dst_lbl), os.path.join(base, 'test.'+src_lbl)
			print("performing TST on {}, src_lbl: {}, dst_lbl: {}".format(fc, fa, fb))
			sa, sb, sc = load_sent(fa), load_sent(fb), load_sent(fc)
			za, zb, zc = encode(sa), encode(sb), encode(sc)
			zd = zc + args.k * (zb.mean(axis=0) - za.mean(axis=0))
			sd = decode(zd)
			output = os.path.join(expd, 'test.'+src_lbl+'to'+dst_lbl)
			write_sent(sd, output)
			# classify test.(src)to(dst)
			sents = load_sent(output)
			preds = classify(sents, classifier_model)
			wrong = 0
			with open(output+'.preds', 'w') as f:
				for i, p in enumerate(preds):
					if(p == (int)(src_lbl)):
						wrong += 1
					f.write(' '.join(sents[i]) + ',' + str(p) + '\n')
				acc = (len(preds) - wrong)/len(preds)
				n_wrong += wrong
				tot += len(preds)
				f.write('Style Transfer Accuracy: {} \n'.format(str(acc)))
		with open(os.path.join(expd, 'tst_acc'), 'w') as f:
			f.write("TST% :{}".format(1 - n_wrong/tot))
	
	if args.sample:
		z = np.random.normal(size=(args.n, model.args.dim_z)).astype('f')
		sents = decode(z)
		write_sent(sents, os.path.join(args.checkpoint, args.output))

	if args.classify:
		sents = load_sent(args.data)
		preds = classify(sents, classifier_model)
		wrong = 0
		with open(args.data+'.preds', 'w') as f:
			for i, p in enumerate(preds):
				if(p == 0):
					wrong += 1
				f.write(' '.join(sents[i]) + '\t' + str(p) + '\n')
			acc = (len(preds) - wrong)/len(preds)
			f.write('Style Transfer Accuracy: {} \n'.format(str(acc)))
	
	if args.clean_dataset: #used to remove outliers from amazon dataset	
		assert args.n == 0 or args.n == 1 #the label each pred should match		
		sents = load_sent(args.data)
		preds = classify(sents, classifier_model)
		with open(args.data+'.cleaned', 'w') as f:
			for i, p in enumerate(preds):
				if(p == int(args.n)):
					line = ' '.join(sents[i])
					f.write(line + '\n')
	
	
	if args.gradientascent:
		assert(isinstance(model, DiversityLSTMClassifier))
		sents = load_sent(args.data)
		base, base_lbls, conv, conv_lbls, attn, masks = gradientascent(sents)
		expd = os.path.join(os.getcwd(), args.checkpoint,'exps_'+args.model_name)
		if(not os.path.exists(expd)):
			os.mkdir(expd)
		with open(os.path.join(expd, 'gradientascent.txt'), 'w') as f:    
			for i in range(len(base)):
				f.write('BASE LBL: ' + str(float(base_lbls[i][1])) + '\n')
				f.write('CONV LBL: ' + str(float(conv_lbls[i][1])) + '\n')
				f.write('BASE SENT: ' + str(base[i])+'\n')
				f.write('BASE ATTN: ' + np_to_str(attn[i]) + '\n')
				f.write('BASE MASK: ' + np_to_str(masks[i]) + '\n')
				f.write('CONV SENT: ' + str(conv[i])+'\n')
				f.write("*"*10 + '\n')

	if args.nlg_eval_on_test:  # produces content preservation metrics on test.1to0 and test.0to1
		args.checkpoint += '/exps_'+args.model_name
		avg_metrics = {}
		lbls = ['0', '1']
		for lbl in lbls:
			dst_lbl = '1' if lbl == '0' else '0'
			path1 = os.path.join(
				args.checkpoint, 'test.{}to{}'.format(lbl, dst_lbl))
			path2 = os.path.join(
				"./data/annotated/{}".format(args.dataset), 'test.{}'.format(lbl))
			print("performing nlg eval between {} and {}".format(path1, path2))
			metrics = compute_metrics(hypothesis=path1, references=[
									  path2], no_skipthoughts=True, not_naturalness=True)
			with open(path1+'.nlg-evals', 'w') as f:
				for k in metrics:
					if(k not in avg_metrics):
						avg_metrics[k] = metrics[k]
					else:
						avg_metrics[k] += metrics[k]
					f.write('{}: {} \n'.format(k, metrics[k]))
		with open(os.path.join(args.checkpoint, 'nlg_evals'), 'w') as f:
			for k in metrics:
				f.write('{}: {} \n'.format(k, avg_metrics[k]/2))

	if args.embedding_tsne:
		expd = os.path.join(os.getcwd(), args.checkpoint,'exps_'+args.model_name)
		if not os.path.exists(expd):
			os.mkdir(expd)
		sents = [[i] for i in range(len(model.vocab.word2idx))]
		english = model.vocab.get_sent(sents)
		input = torch.tensor(sents, device = model.embed.weight.data.device)
		model.encode(input, args=None)
		z = model.z.squeeze().detach().cpu().numpy()
		tsne = TSNE(n_jobs=10)
		res = tsne.fit_transform(z)
		print("Done performing TSNE, Plotting...")
		plt.figure(figsize=(15, 15))
		for i in tqdm(range(args.m)):
			plt.text(res[i,0], res[i,1], english[i], fontsize=7, rotation=30)
		plt.savefig(os.path.join(expd, 'tsne-embd.png'))

	if args.tsne:
		sents = load_sent(args.data)
		z, l = encode_annotated(sents)
		tsne = TSNE(n_jobs=10)
		res = tsne.fit_transform(z)
		color_map = np.argmax(l, axis=1)
		plt.figure(figsize=(10, 10))
		for cl in range(2):
			indices = np.where(color_map == cl)[0]
			plt.scatter(res[indices, 0], res[indices, 1], label=cl)
		plt.legend()
		plt.savefig('./img/tsne.png')

	if args.reconstruct:
		sents = load_sent(args.data)
		z = encode(sents)
		sents_rec = decode(z)
		write_z(z, os.path.join(args.checkpoint, args.output+'.z'))
		write_sent(sents_rec, os.path.join(
			args.checkpoint, args.output+'.rec'))
	
	if args.cond_reconstruct_ae: 
		data = load_sent_mt(args.data)
		z, base = encode_mt(data)
		assert(len(z) == len(base))
		sents_rec = decode_mt(z, base, cond_recon=True)
		write_z(z, os.path.join(args.checkpoint, args.output+'.z'))
		write_sent(sents_rec, os.path.join(args.checkpoint, args.output+'.rec'))

	if args.arithmetic:
		fa, fb, fc = args.data.split(',')
		sa, sb, sc = load_sent(fa), load_sent(fb), load_sent(fc)
		za, zb, zc = encode(sa), encode(sb), encode(sc)
		zd = zc + args.k * (zb.mean(axis=0) - za.mean(axis=0))
		sd = decode(zd)
		write_sent(sd, os.path.join(args.checkpoint, args.output))

	if args.bleu:
		def load_sent(path):
			s1, s2 = [], []
			with open(path, 'r') as f:
				for line in f:
					a, b = line.split(',')[0], line.split(',')[1]
					s1.append(a)
					s2.append(b)
			return s1, s2
		base, conv = load_sent(args.data)
		assert(len(base) == len(conv))
		mean_bleu_score = 0.0
		f = nltk.translate.bleu_score.SmoothingFunction(0.000000001).method1
		for i in range(len(base)):
			mean_bleu_score += sentence_bleu([base[i]], conv[i], smoothing_function=f)
		print(mean_bleu_score/len(base))

	if args.interpolate:
		f1, f2 = args.data.split(',')
		s1, s2 = load_sent(f1), load_sent(f2)
		z1, z2 = encode(s1), encode(s2)
		zi = [interpolate(z1_, z2_, args.n) for z1_, z2_ in zip(z1, z2)]
		zi = np.concatenate(zi, axis=0)
		si = decode(zi)
		si = list(zip(*[iter(si)]*(args.n)))
		write_doc(si, os.path.join(args.checkpoint, args.output))

	if args.latent_nn:
		sents = load_sent(args.data)
		z = encode(sents)
		with open(os.path.join(args.checkpoint, args.output), 'w') as f:
			nn = NearestNeighbors(n_neighbors=args.n).fit(z)
			dis, idx = nn.kneighbors(z[:args.m])
			for i in range(len(idx)):
				f.write(' '.join(sents[i]) + '\n')
				for j, d in zip(idx[i], dis[i]):
					f.write(' '.join(sents[j]) + '\t%.2f\n' % d)
				f.write('\n')

	if args.kmeans:
		sents = load_sent(args.data)
		z = encode(sents)
		with open(os.path.join(args.checkpoint, args.output), 'w') as f:
			km = KMeans(n_clusters=args.n).fit(z)
			dis, idx = nn.kneighbors(z[:args.m])
			for i in range(len(idx)):
				f.write(' '.join(sents[i]) + '\n')
				for j, d in zip(idx[i], dis[i]):
					f.write(' '.join(sents[j]) + '\t%.2f\n' % d)
				f.write('\n')
	
	if args.style_masking: #convert an annotated dataset into style masked dataset
		def get_bleu(dpath):
			def load_sent(path):
				s1, s2 = [], []
				with open(path, 'r') as f:
					for line in f:
						a, b = line.split(',')[0], line.split(',')[1]
						s1.append(a)
						s2.append(b)
				return s1, s2
			base, conv = load_sent(dpath)
			assert(len(base) == len(conv))
			mean_bleu_score = 0.0
			f = nltk.translate.bleu_score.SmoothingFunction(0.000000001).method1
			for i in range(len(base)):
				mean_bleu_score += sentence_bleu([base[i]], conv[i], smoothing_function=f)
			print("bleu of {}:{}".format(dpath,mean_bleu_score/len(base)))

		assert(args.dataset != None)
		base = './data/annotated/{}'.format(args.dataset)
		if(args.data is not None):
			args.dataset += '-' + args.data
		conv = './data/style-masked/{}'.format(args.dataset)
	
		if(args.ig or args.grads): #for other types of attr, we only need test data
			files = ['test']
		else:
			files = ['test','train','valid']

		for f in files:
			path = os.path.join(base, f+'.csv')
			path2 = os.path.join(conv, f+'.csv')
			print("Creating {}".format(path2))
			sents = load_sent(path)
			inputs, targets, lbls = style_masking(sents, model)
			lbl_split = lbl_wise_split(inputs, targets, lbls)
			inputs, targets = model.vocab.get_sent(inputs), model.vocab.get_sent(targets)
			if(not os.path.exists(conv)): #create dirs if needed
				os.mkdir(conv)
				os.mkdir(conv+'/0')
				os.mkdir(conv+'/1')
			with open(path2, 'w') as fi: #create {split.csv} with all labels
				for i in range(len(inputs)):
					assert(len(inputs[i].split()) == len(targets[i].split())) #they have same #tokens
					fi.write(inputs[i] + ' , ' + targets[i] + ' , ' + str(lbls[i]) + '\n')
			for lbl in lbl_split: #for each label make separate style masked dataset
				conv_lbl = './data/style-masked/{}/{}'.format(args.dataset, lbl)
				print(conv_lbl)
				print("Creating {}".format(f))
				with open(os.path.join(conv_lbl, f+'.csv'), 'w') as fi2: #create {split}.csv
					inps, tars = lbl_split[lbl]['inputs'], lbl_split[lbl]['targets']
					inps, tars = model.vocab.get_sent(inps), model.vocab.get_sent(tars)
					for i in range(len(inps)):
						fi2.write(inps[i] + ' , ' + tars[i] + ' , ' + lbl + '\n')
				with open(os.path.join(conv_lbl, f+'.txt'), 'w') as fi3: #create {split}.txt (unannotated)
					inps, tars = lbl_split[lbl]['inputs'], lbl_split[lbl]['targets']
					inps, tars = model.vocab.get_sent(inps), model.vocab.get_sent(tars)
					for i in range(len(inps)):
						fi3.write(tars[i] + '\n')
				# get_bleu(os.path.join(conv_lbl, f+'.csv'))
	
	if args.mt_tst_on_test_ae: #tst exp for style masked dataset
		assert args.data != None
		assert args.classifier_checkpoint != None
		#args.data -> directory that contains individual autoencoder models, one for each style
		lbls = [str(lbl) for lbl in range(args.output_size)]
		models = {lbl: [os.path.join(args.data, str(lbl)), None, None] for lbl in lbls} #lbl -> (model path, model object, model vocab)
		data = {lbl: os.path.join('./data/style-masked',args.dataset, lbl, 'test.csv') for lbl in lbls} #one autoencoder for each style
		for lbl in lbls:
			ckpt = models[lbl][0]
			models[lbl][2] = Vocab(os.path.join(ckpt, 'vocab.txt'))
			models[lbl][1], train_args = get_model(os.path.join(ckpt, 'checkpoints', args.model_name), models[lbl][2])
		#perform TST from src_lbl->dst_lbl
		corr, tot = 0, 0
		base, conv = [], []
		for src_lbl in range(args.output_size):
			for dst_lbl in range(args.output_size):
				if(src_lbl == dst_lbl):
					continue
				d = data[str(src_lbl)]
				src_data = load_sent_mt(d)
				model = models[str(dst_lbl)][1]
				vocab = models[str(dst_lbl)][2]
				z, _ = encode_mt(src_data)
				sents_rec = decode(z)
				preds = classify(sents_rec, classifier_model)
				corr += len(preds[preds==dst_lbl])
				tot += len(z)
				acc = len(preds[preds==dst_lbl])/len(z)
				base.extend(src_data[0])
				conv.extend(sents_rec)
				# sents_rec.append(['TST%:', str(acc)])
				expd = os.path.join(os.getcwd(), args.data, str(dst_lbl),'exps_'+args.model_name)
				if not os.path.exists(expd):
					os.mkdir(expd)
				write_sent(sents_rec, os.path.join(expd,'test.{}to{}'.format(src_lbl, dst_lbl)))
		assert(len(base) == len(conv))
		with open(os.path.join(args.data, 'tst_acc.txt'), 'w') as f:
			for i in range(len(base)):
				f.write(' '.join(base[i]) + ' , ' + ' '.join(conv[i]) + '\n')
			f.write('TST%: {}'.format(corr/tot))
	
	if args.mt_nlg_eval_on_test:  # nlg evals for tst using style masked dataset
		assert args.data != None
		avg_metrics = {}
		lbls = ['0', '1']
		for src_lbl in lbls:
			dst_lbl = '1' if src_lbl == '0' else '0'
			args.checkpoint = os.path.join(os.path.join(args.data, dst_lbl), 'exps_'+args.model_name)
			path1 = os.path.join(args.checkpoint, 'test.{}to{}'.format(src_lbl, dst_lbl))
			path2 = os.path.join("./data/style-masked/{}/{}".format(args.dataset, src_lbl), 'test.txt'.format(src_lbl))
			print("performing nlg eval between {} and {}".format(path1, path2))
			metrics = compute_metrics(hypothesis=path1, references=[path2], no_skipthoughts=True, not_naturalness=True)
			with open(path1+'.nlg-evals', 'w') as f:
				for k in metrics:
					if(k not in avg_metrics):
						avg_metrics[k] = metrics[k]
					else:
						avg_metrics[k] += metrics[k]
					f.write('{}: {} \n'.format(k, metrics[k]))
		with open(os.path.join(args.data, 'nlg_evals'), 'w') as f:
			for k in metrics:
				f.write('{}: {} \n'.format(k, avg_metrics[k]/2))
	
	if args.cond_reconstruct_tr: #cond reconstruct on style masked dataset using transformer
		def write_sent(sents, path):
			with open(path, 'w') as f:
				for s in sents:
					f.write(s+'\n')
		data = load_sent_mt(args.data)
		batches, order = get_batches_mt(data, vocab, args.batch_size, device, sort=False)
		sents_rec, base = [], []
		for inputs, targets, lbls in batches:
			outs = model(inputs, train_args, is_train=False).argmax(dim=-1).t().tolist() #BL
			sents_rec.extend(strip_eos(model.vocab.get_sent(outs)))    
		write_sent(sents_rec, os.path.join(args.checkpoint, args.output+'.rec'))
	
	if args.cond_reconstruct_tr_single: #cond reconstruct on style masked dataset using transformer
		def write_sent(sents, path):
			with open(path, 'w') as f:
				for s in sents:
					f.write(s+'\n')
		data = load_sent_mt(args.data)
		batches, order = get_batches_mt(data, vocab, args.batch_size, device, sort=False)
		sents_rec, base = [], []	
		for inputs, targets, lbls in batches:		
			dst_lbls = torch.logical_not(lbls).to(torch.long)	# only works for 2 lbls
			inputs, _ = model.concat_meta(inputs, targets, dst_lbls)
			outs = model(inputs, train_args, is_train=False)[1].argmax(dim=-1).t().tolist() #BL
			sents_rec.extend(strip_eos(model.vocab.get_sent(outs)))    
		write_sent(sents_rec, os.path.join(args.checkpoint, args.output+'.rec'))
			
	if args.mt_tst_on_test_tr: #tst exp on sm datasets using 2 transformer
		assert args.data != None
		assert args.classifier_checkpoint != None
		#args.data -> directory that contains individual autoencoder models, one for each style
		lbls = [str(lbl) for lbl in range(args.output_size)]
		models = {lbl: [os.path.join(args.data, str(lbl)), None, None] for lbl in lbls} #lbl -> (model path, model object, model vocab)
		data = {lbl: os.path.join('./data/style-masked',args.dataset, lbl, 'test.csv') for lbl in lbls} #one autoencoder for each style
		for lbl in lbls:
			ckpt = models[lbl][0]
			models[lbl][2] = Vocab(os.path.join(ckpt, 'vocab.txt'))
			models[lbl][1], train_args = get_model(os.path.join(ckpt, 'checkpoints', args.model_name), models[lbl][2])
		#perform TST from src_lbl->dst_lbl
		corr, tot = 0, 0
		base, conv = [], []
		for src_lbl in range(args.output_size):
			for dst_lbl in range(args.output_size):
				if(src_lbl == dst_lbl):
					continue
				d = data[str(src_lbl)]
				src_data = load_sent_mt(d)
				model = models[str(dst_lbl)][1]
				vocab = models[str(dst_lbl)][2]
				sents_rec = decode_mt_tr(src_data, model)
				preds = classify(sents_rec, classifier_model)
				assert(len(preds) == len(sents_rec))
				# sents_rec = [sents_rec[i] + [' , '] + preds[i] for i in range(len(preds))]
				corr += len(preds[preds==dst_lbl])
				tot += len(sents_rec)
				acc = len(preds[preds==dst_lbl])/len(sents_rec)
				base.extend(src_data[0])
				conv.extend(sents_rec)
				# sents_rec.append(['TST%:',str(acc)])
				expd = os.path.join(os.getcwd(), args.data, str(dst_lbl),'exps_'+args.model_name)
				if not os.path.exists(expd):
					os.mkdir(expd)
				write_sent(sents_rec, os.path.join(expd,'test.{}to{}'.format(src_lbl, dst_lbl)))
		assert(len(base) == len(conv))
		with open(os.path.join(args.data, 'tst_acc.txt'), 'w') as f:
			for i in range(len(base)):
				f.write(' '.join(base[i]) + ' , ' + ' '.join(conv[i]) + '\n')
			f.write(str(corr/tot))
	
	if args.mt_tst_on_test_tr_single: #tst exp on sm datasets using single transformer
		assert args.classifier_checkpoint != None
		lbls = [str(lbl) for lbl in range(args.output_size)]
		data = {lbl: os.path.join('./data/style-masked',args.dataset, lbl, 'test.csv') for lbl in lbls}
		corr, tot = 0, 0
		base, conv = [], []
		for src_lbl in range(args.output_size):
			for dst_lbl in range(args.output_size):
				if(src_lbl == dst_lbl):
					continue
				src_data = load_sent_mt(data[str(src_lbl)])
				sents_rec = decode_mt_tr_single(src_data, model)
				preds = classify(sents_rec, classifier_model)
				assert(len(preds) == len(sents_rec))
				corr += len(preds[preds==dst_lbl])
				tot += len(sents_rec)
				acc = len(preds[preds==dst_lbl])/len(sents_rec)
				base.extend(src_data[0])
				conv.extend(sents_rec)
				# sents_rec.append(['TST%:',str(acc)])
				expd = os.path.join(os.getcwd(), args.checkpoint, 'exps_'+args.model_name)
				if not os.path.exists(expd):
					os.mkdir(expd)
				write_sent(sents_rec, os.path.join(expd,'test.{}to{}'.format(src_lbl, dst_lbl)))
		assert(len(base) == len(conv))
		with open(os.path.join(expd, 'tst_acc.txt'), 'w') as f:
			for i in range(len(base)):
				f.write(' '.join(base[i]) + ' , ' + ' '.join(conv[i]) + '\n')
			f.write(str(corr/tot))
	
	if args.mt_nlg_eval_on_test_tr_single:  # nlg evals for tst using style masked dataset
		assert args.data != None
		args.checkpoint = os.path.join(args.data, 'exps_'+args.model_name)
		avg_metrics = {}
		lbls = ['0', '1']
		for src_lbl in lbls:
			dst_lbl = '1' if src_lbl == '0' else '0'	
			path1 = os.path.join(args.checkpoint, 'test.{}to{}'.format(src_lbl, dst_lbl))
			path2 = os.path.join("./data/style-masked/{}/{}".format(args.dataset, src_lbl), 'test.txt'.format(src_lbl))
			print("performing nlg eval between {} and {}".format(path1, path2))
			metrics = compute_metrics(hypothesis=path1, references=[path2], no_skipthoughts=True, not_naturalness=True)
			with open(path1+'.nlg-evals', 'w') as f:
				for k in metrics:
					if(k not in avg_metrics):
						avg_metrics[k] = metrics[k]
					else:
						avg_metrics[k] += metrics[k]
					f.write('{}: {} \n'.format(k, metrics[k]))
		with open(os.path.join(args.checkpoint, 'nlg_evals'), 'w') as f:
			for k in metrics:
				f.write('{}: {} \n'.format(k, avg_metrics[k]/2))
	
	if args.create_forms:
		def get_sents(path):
			sents = []
			with open(path, 'r') as f:
				sents = f.readlines()
			return sents

		p1, p2, p3 = args.data.split(',')[0], args.data.split(',')[
			1], args.data.split(',')[2]
		sents1, sents2, sents3 = get_sents(p1), get_sents(p2), get_sents(p3)
		assert len(sents1) == len(sents2) == len(sents3)
		for i in range(0, 100, 20):
			cnt = 0
			with open(args.output+str(i/20), 'w') as f:
				for j in range(i, i+20, 1):
					f.write("{}. \n base: {}  model1: {}  model2: {} Decision: \n".format(
						j, sents3[j], sents1[j], sents2[j]))
				cnt += 1