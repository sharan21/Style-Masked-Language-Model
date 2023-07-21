import torch
import torch.nn.functional as F
import numpy as np

# class Batchify:
#     self.constant_batch=False
#     self.sort=False
#     self.no_stokens=False
#     self.dataset_type='unlabelled'

#     def __init__(self, args, vocab, device, constant_batch, sort, no_stokens):
#         pass
    
#     def get_batches(self):
#         pass

#     def get_batch(self):
#         pass

def get_batch(x, vocab, device):
    go_x, x_eos = [], []
    max_len = max([len(s) for s in x])
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        go_x.append([vocab.go] + s_idx + padding)
        x_eos.append(s_idx + [vocab.eos] + padding)
    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch

def get_batches(data, vocab, batch_size, device, constant_batch=False, sort=True): #used for loading sents without labels
    order = range(len(data))
    if(sort): # sort in increasing order of sentence length
        z = sorted(zip(order, data), key=lambda i: len(i[1]))
        order, data = zip(*z)
    batches = []
    i = 0
    if(constant_batch): #all batches will have leading lenght = batch_size
        while(i + batch_size < len(data)):
            batches.append(get_batch(data[i: i+batch_size], vocab, device))    
            i += batch_size
    else: #all instances in batch of same size
        while i < len(data):
            j = i
            while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
                j += 1
            batches.append(get_batch(data[i: j], vocab, device))
            i = j
    return batches, order


def get_batch_annotated(x, vocab, device, num_classes=2, no_stokens=False):
    go_x, x_eos, targets = [], [], []
    max_len = max([len(s) for s in x])
    for i, s in enumerate(x):
        s_idx = [vocab.word2idx[s[i]] if s[i] in vocab.word2idx else vocab.unk for i in range(len(s) - 2)]
        padding = [vocab.pad] * (max_len - len(s))
        if(no_stokens):
            go_x.append(s_idx + padding)
            x_eos.append(s_idx + padding)
        else:
            go_x.append([vocab.go] + s_idx + padding)
            x_eos.append(s_idx + [vocab.eos] + padding)
        targets.append((int)(s[-1]))    
    targets = F.one_hot(torch.tensor([targets]), num_classes=num_classes).to(torch.float).squeeze()
    if(len(targets.shape) == 1):
        targets = targets.unsqueeze(dim=0)
    return torch.LongTensor(go_x).t().contiguous().to(device), torch.LongTensor(x_eos).t().contiguous().to(device),  targets.contiguous().to(device)

def get_batches_annotated(data, vocab, batch_size, device, num_classes=2, sort=True, no_stokens=False):
    order = range(len(data))
    if(sort): # sort in increasing order of sentence length
        z = sorted(zip(order, data), key=lambda i: len(i[1]))
        order, data = zip(*z)   
    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        batches.append(get_batch_annotated(data[i: j], vocab, device, num_classes=num_classes, no_stokens=no_stokens))
        i = j
    return batches , order


def get_batches_mt(data, vocab, batch_size, device, num_classes=2, sort=True):#data -> (inputs, targets, lbls)
    inputs, targets, lbls = data    
    n = len(inputs)
    order = range(n)
    if(sort): # sort in increasing order of sentence length
        z = sorted(zip(order, inputs, targets, lbls), key=lambda i: len(i[1]))
        order, inputs, targets, lbls = zip(*z)   
    batches = []
    i = 0
    assert len(inputs) == len(targets) == len(lbls)
    while i < n:
        j = i
        while j < min(n, i+batch_size) and len(inputs[i]) == len(inputs[j]):
            j += 1
        batches.append(get_batch_mt(inputs[i: j], targets[i:j], lbls[i:j], vocab, device, num_classes=num_classes))
        i = j
    return batches , order

def get_batch_mt(inps, tar, lbl, vocab, device, num_classes=2): #x is a batch
    inputs, targets, lbls = [], [], []
    assert(len(inps) == len(tar) == len(lbl))
    max_len = max([len(s) for s in inps]) #of this batch
    for i in range(len(inps)): #iterate across batch 
        t1 = [vocab.word2idx[inps[i][y]] if inps[i][y] in vocab.word2idx else vocab.unk for y in range(len(inps[i]))]
        t2 = [vocab.word2idx[tar[i][y]] if tar[i][y] in vocab.word2idx else vocab.unk for y in range(len(tar[i]))]
        padding = [vocab.pad] * (max_len - len(inps[i]))  
        inputs.append([vocab.go] + t1 + padding)
        targets.append(t2 + [vocab.eos] + padding)
        lbls.append((int)(lbl[i]))    
    lbls = F.one_hot(torch.tensor([lbls]), num_classes=num_classes).to(torch.float).squeeze()
    if(len(lbls.shape) == 1):
        lbls = lbls.unsqueeze(dim=0)
    return torch.LongTensor(inputs).t().contiguous().to(device), torch.LongTensor(targets).t().contiguous().to(device), lbls.contiguous().to(device)

