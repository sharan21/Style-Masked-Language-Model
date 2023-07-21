
import random
import os
import math
import collections
import numpy as np
import torch
import pickle
from vocab import Vocab

class ClassifierEvaluationDataset(torch.utils.data.Dataset):
	def __init__(self, encodings):
		self.encodings = encodings

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		return item

	def __len__(self):
		return len(self.encodings.input_ids)

class AverageMeter(object):
	def __init__(self):
		self.clear()

	def clear(self):
		self.cnt = 0
		self.sum = 0
		self.avg = 0

	def update(self, val, n=1):
		self.cnt += n
		self.sum += val * n
		self.avg = self.sum / self.cnt

class NearestTokenLookUp(torch.nn.Module):
	def __init__(self, ortho_model, args=None):
		super().__init__()
		self.ortho_model = ortho_model 
		self.all_z = [] # list of all latent vectors
		self.hash2idx = {} # hash to token
		self.build(args)

	def get_nearest_z(self, z):
		b, l, dim_z = z.shape
		z = np.reshape(z, (-1, z.shape[-1])) # BLZ -> B*L, Z
		_, idx = self.nn.kneighbors(z)
		nearest_z = self.all_z[idx]
		nearest_z = np.reshape(nearest_z, (b, l, dim_z)) # B*L,Z -> BLZ
		return nearest_z

	def z2idx(self, z): #get the idx of a particular latent vector
		h = hash(tuple(z))
		assert h in self.hash2idx
		return self.hash2idx[h]

	def get_sents_from_z(self, z):
		b, l, dim_z = z.shape #BLZ
		z = np.reshape(z, (-1, dim_z)) #B*L,Z
		idxs = []
		for z_ in z:
			idxs.append(self.z2idx(z_))
		idxs = np.array(idxs) #B*L
		idxs = np.reshape(idxs, (b, l)) #BL
		return idxs
		
	def build(self, args):
		#get all token idx
		self.idx = torch.tensor(list(range(len(self.ortho_model.vocab.word2idx))), device=self.ortho_model.embed.weight.data.device)
		#get latent z of all tokens
		self.ortho_model.encode(self.idx, args=args)
		self.all_z = self.ortho_model.z.detach().cpu().numpy()
		#train nn model on all latent z
		self.nn = NearestNeighbors(n_neighbors=1).fit(self.all_z)
		# build z -> idx dicts
		for idx, z in enumerate(self.all_z):
			self.hash2idx[hash(tuple(z))] = idx

def evaluate(model, batches, args):
	model.eval()
	meters = collections.defaultdict(lambda: AverageMeter())
	with torch.no_grad():
		for inputs, targets in batches:
			losses = model.autoenc(inputs, targets, args)
			for k, v in losses.items():
				meters[k].update(v.item(), inputs.size(1))
	loss = model.loss({k: meter.avg for k, meter in meters.items()})
	meters['loss'].update(loss)
	return meters
	
def shuffle_in_unison(a, b):
	c = list(zip(a, b))
	random.shuffle(c)
	a, b = zip(*c)
	return a, b

def get_anneal_weight(step, args):
		assert(args.fn != 'none')
		if args.fn == 'logistic':
			return 1 - float(1/(1+np.exp(-args.k*(step-args.tot_steps))))    
		elif args.fn == 'sigmoid':
			return 1 - (math.tanh((step - args.tot_steps * 1.5)/(args.tot_steps / 3))+ 1)
		elif args.fn == 'linear': 
			return min(1, step/args.tot_steps)
		else:
			exit("wrong option in args.fn")

def clean_line(line, bad=[',','.', ';', '(', ')', '/', '`', '%', '"', '-', '\\','\'','?']): # use this function carefully
	clean = ''
	for c in line:
		if c not in bad:
			clean += c
	return clean

def set_seed(seed):     # set the random seed for reproducibility
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

def create_glove(embd_dim=300):
	path = 'glove/glove.6B.{}d.txt'.format(embd_dim)
	print("Creating glove embds from {}".format(path))
	glove = {}    
	with open(path, 'rb') as f:
		for idx, l in enumerate(f):
			if(idx % 100000 == 0):
				print(idx)
			line = l.decode().split()
			glove[line[0]] = np.array(line[1:]).astype(np.float)
	pickle.dump(glove, open('glove/glove-{}.pkl'.format(embd_dim), 'wb'))

def load_glove(embd_dim=300):
	path = 'glove/glove-{}.pkl'.format(embd_dim)
	if(os.path.exists(path)):
		glove = pickle.load(open(path, 'rb'))
	else:
		exit("did not find glove in given path")
	return glove

def get_glove(vocab, embd_dim=300):
	glove = load_glove(embd_dim)
	glove_mat = np.zeros((len(vocab.word2idx), glove[next(iter(glove))].size))
	for word, idx in vocab.word2idx.items():
		glove_mat[idx] = glove[word] if word in glove else np.random.normal(scale=0.6, size=(glove[next(iter(glove))].size, ))
	return glove_mat

def strip_eos(sents):
	return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
		for sent in sents]

def load_sent(path): # used to load sents from unannotated datasets with format: {input}
	inputs = []
	with open(path) as f:
		for line in f:
			inputs.append(line.split())
	return inputs

def load_sent_anno(path): # used to load sents from annotated datasets with format: {input} , {lbl}
	inputs, lbls = [], []
	with open(path) as f:
		for line in f:
			line = line.split(',')
			inp, lbl = line[0].split(), line[1].split()
			inputs.append(inp)
			lbls.append(lbl)
	return inputs, lbls

def load_sent_mt(path): # used to load sents from style-masked datasets with format: {input}, {target}, {lbl}
	inputs, targets, lbls = [], [], []
	bad = 0
	with open(path) as f:
		for i, line in enumerate(f):
			line = line.split(',')
			inp, tar, lbl = line[0].split(), line[1].split(), line[2]
			if len(inp) != len(tar) or (not inp or not tar): #incase any sentence is blank
				bad += 1
				continue
			inputs.append(inp)
			targets.append(tar)
			lbls.append(lbl)
	assert(bad == 0)
	return inputs, targets, lbls

def write_sent(sents, path):
	with open(path, 'w') as f:
		for s in sents:
			l = ' '.join(s) + '\n'
			if l == '\n':
				f.write('<blank> \n')	
			else:
				f.write(l)

def write_doc(docs, path):
	with open(path, 'w') as f:
		for d in docs:
			for s in d:
				f.write(' '.join(s) + '\n')
			f.write('\n')

def write_z(z, path):
	with open(path, 'w') as f:
		for zi in z:
			for zij in zi:
				f.write('%f ' % zij)
			f.write('\n')

def logging(s, path, print_=True):
	if print_:
		print(s)
	if path:
		with open(path, 'a+') as f:
			f.write(s + '\n')
	
def lerp(t, p, q):
	return (1-t) * p + t * q

# spherical interpolation https://github.com/soumith/dcgan.torch/issues/14#issuecomment-199171316
def slerp(t, p, q):
	o = np.arccos(np.dot(p/np.linalg.norm(p), q/np.linalg.norm(q)))
	so = np.sin(o)
	return np.sin((1-t)*o) / so * p + np.sin(t*o) / so * q

def interpolate(z1, z2, n):
	z = []
	for i in range(n):
		zi = lerp(1.0*i/(n-1), z1, z2)
		z.append(np.expand_dims(zi, axis=0))
	return np.concatenate(z, axis=0)

def shuffle_two_lists(list1, list2):
	temp = list(zip(list1, list2))
	random.shuffle(temp)
	res1, res2 = zip(*temp)
	return res1, res2

def flip(p=0.008):
	return True if random.random() < p else False

def np_to_str(var): #converts numpy object to string
	return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

def get_masks_from_attn(attn, eps=0):
	l = attn.shape[0]
	base = ((1+eps)/l) * torch.ones_like(attn) #uniform dist
	mask = torch.zeros_like(attn)
	mask[attn > base] = 1
	return mask.unsqueeze(dim=-1) #L1

def get_masks_from_attn_batch(attn, eps):
	if(len(attn.shape) == 1): #this happens sometimes when B=1
		attn = attn.unsqueeze(dim=0)
	b, l = attn.shape
	base = ((1+eps)/l) * torch.ones_like(attn) #uniform dist
	mask = torch.zeros_like(attn)
	mask[attn > base] = 1
	return mask #BL

def lbl_wise_split(inputs, targets, lbls):
	split = {}
	for i, lbl in enumerate(lbls):
		if(str(lbl) not in split):
			split[str(lbl)] = {'inputs': [], 'targets':[]}
		split[str(lbl)]['inputs'].append(inputs[i])
		split[str(lbl)]['targets'].append(targets[i])
	for k in split:
		assert len(split[k]['inputs']) == len(split[k]['targets'])
	return split

def compute_acc(yhat, targets):
	yhat = yhat.squeeze()
	targets = targets.squeeze()
	n_samples = targets.size(0)
	yhat = torch.argmax(yhat, dim=-1)
	targets = torch.argmax(targets, dim=-1)    
	return yhat[yhat == targets].size(0)/n_samples
