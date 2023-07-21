import pandas as pd
import random
import jsonlines
from tqdm import tqdm
import json
import pandas as pd
from utils import shuffle_in_unison, clean_line

def split_testset(path): # splits test.csv into test.label1, test.label2...
    d = {}
    with open(path, 'r') as f:
        for line in f:
            s, l = line.split(',')[0], line[-2]
            if l not in d:
                d[l] = []
            d[l].append(s)
    for k in d:
        with open(path+'.'+k, 'w') as f:
            for line in d[k]:
                f.write(line+'\n')
  
def split_dataset(sents, splits=[0.8, 0.1, 0.1]):
    trainl, validl = (int)(len(sents)*splits[0]), (int)(len(sents)*(splits[0]+splits[1]))
    print("tr, va, te: {}, {}, {}".format(trainl, validl-trainl, len(sents)-validl))
    return trainl, validl
    
def create_tenses_dataset():
    def write_to_disk(sents, path):
        with open(path, "w") as f:
            for s in sents:
                f.write(s+'\n')
    def write_to_disk2(sents, labels, path):
        with open(path, "w") as f:
            for i in range(len(sents)):
                line = sents[i] + " , " + str(labels[i]) + "\n"
                f.write(line)

    def preproc_tense(path):
        sents, labels, past, present, future = [], [], [], [], []
        with open(path, 'r') as f:
            for line in f:
                try:
                    lbl = int(line[0]) - 1
                except:
                    continue
                if(lbl == -1):
                    continue
                t = line.split('\t')
                t = clean_line(t[-1][:-1])
                if(lbl == 0):
                    future.append(t)
                elif(lbl == 1):
                    past.append(t)
                else:
                    present.append(t)
                sents.append(t)
                labels.append(lbl)
        return sents, labels, past, present, future    

    base = './data/style-ptb-annotated/'
    d = ['tense-ppr', 'tense-voice', 'tense-ppfb','tense-voice-ppr']
    splits = ['/train.tsv', '/test.tsv', '/valid.tsv']
    sents, labels, past, present, future = [], [], [], [], []
    for d_ in d:
        for split in splits:
            path = base + d_ + split
            print("importing {}".format(path))
            s, l, pa, pr, fu = preproc_tense(path)
            sents.extend(s)
            labels.extend(l)
            past.extend(pa)
            present.extend(pr)
            future.extend(fu)
    assert(len(sents) == len(labels))
    labels, sents = shuffle_in_unison(labels, sents)    
    trainl, validl = split_dataset(sents)
    print("#sentences: {}".format(len(sents)))
    print("pa, pr, fu: {}, {}, {}".format(pa, pr, fu))
    #annotated
    write_to_disk2(sents, labels, "./data/style-ptb-annotated/tenses/tenses.csv")
    write_to_disk2(sents[:trainl], labels[:trainl], "./data/style-ptb-annotated/tenses/train.csv")
    write_to_disk2(sents[trainl:validl], labels[trainl:validl], "./data/style-ptb-annotated/tenses/valid.csv")
    write_to_disk2(sents[validl:], labels[validl:], "./data/style-ptb-annotated/tenses/test.csv")
    #non-annotated
    write_to_disk(sents[:trainl], "./data/style-ptb/tenses/train.txt")
    write_to_disk(sents[trainl:validl], "./data/style-ptb/tenses/valid.txt")
    write_to_disk(sents[validl:], "./data/style-ptb/tenses/test.txt")
    #label wise
    write_to_disk(list(set(past)), "./data/style-ptb/tenses/past.txt")
    write_to_disk(list(set(present)), "./data/style-ptb/tenses/present.txt")
    write_to_disk(list(set(future)), "./data/style-ptb/tenses/future.txt") 

def create_ppr_dataset():
    def write_to_disk(sents, path):
        with open(path, "w") as f:
            for s in sents:
                f.write(s+'\n')
    def write_to_disk2(sents, labels, path):
        with open(path, "w") as f:
            for i in range(len(sents)):
                line = sents[i] + " , " + str(labels[i]) + "\n"
                f.write(line)
                
    def preproc_ppr(path, idx=0):
        sents, labels, ppr, no_ppr = [], [], [], []
        with open(path, 'r') as f:
            for line in f:
                try:
                    lbl = int(line[idx])
                except:
                    continue
                t = line.split('\t')
                t = clean_line(t[-1][:-1])
                if(lbl == 0):
                    ppr.append(t)
                else:
                    no_ppr.append(t)
                labels.append(lbl)
                sents.append(t)
        return sents, labels, ppr, no_ppr    
    base = './data/style-ptb-annotated/'
    d = [ 'tense-ppr', 'ppr-voice','tense-voice-ppr']
    splits = ['/train.tsv', '/test.tsv', '/valid.tsv']
    sents, labels, ppr, no_ppr = [], [], [], []
    for d_ in d:
        for split in splits:
            path = base + d_ + split
            print("importing from {}".format(path))
            if(d_ == 'tense-ppr'):
                s, l, ppr_, no_ppr_ = preproc_ppr(path, idx=2)
            elif(d_ == 'ppr-voice'):
                s, l, ppr_, no_ppr_ = preproc_ppr(path, idx=0)
            elif('tense-voice-ppr'):
                s, l, ppr_, no_ppr_ = preproc_ppr(path, idx=4)
            ppr.extend(ppr_)
            no_ppr.extend(no_ppr_)
            sents.extend(s)
            labels.extend(l)

    assert(len(sents) == len(labels))
    labels, sents = shuffle_in_unison(labels, sents)    
    trainl, validl = split_dataset(sents)
    print("#sentences: {}".format(len(sents)))
    #annotated
    write_to_disk2(sents, labels, "./data/style-ptb-annotated/ppr/ppr.csv")
    write_to_disk2(sents[:trainl], labels[:trainl], "./data/style-ptb-annotated/ppr/train.csv")
    write_to_disk2(sents[trainl:validl], labels[trainl:validl], "./data/style-ptb-annotated/ppr/valid.csv")
    write_to_disk2(sents[validl:], labels[validl:], "./data/style-ptb-annotated/ppr/test.csv")
    #non annotated
    write_to_disk(sents[:trainl], "./data/style-ptb/ppr/train.txt")
    write_to_disk(sents[trainl:validl], "./data/style-ptb/ppr/valid.txt")
    write_to_disk(sents[validl:], "./data/style-ptb/ppr/test.txt")
    #label wise
    write_to_disk(list(set(ppr)), "./data/style-ptb/ppr/has-ppr.txt")
    write_to_disk(list(set(no_ppr)), "./data/style-ptb/ppr/no-ppr.txt")
    
def create_voices_dataset():
    def write_to_disk(sents, path):
        with open(path, "w") as f:
            for s in sents:
                f.write(s+'\n')
    def write_to_disk2(sents, labels, path):
        with open(path, "w") as f:
            for i in range(len(sents)):
                line = sents[i] + " , " + str(labels[i]) + "\n"
                f.write(line)
    def preproc_voice(path, idx=0):
        sents, labels, active, passive = [], [], [], []
        with open(path, 'r') as f:
            for line in f:
                try:
                    lbl = int(line[idx]) - 1
                except:
                    continue
                if(lbl == -1):
                    continue
                t = line.split('\t')
                l = clean_line(t[-1][:-1])
                sents.append(l)
                labels.append(lbl)
                if(lbl == 0):
                    passive.append(l)
                    
                else:
                    active.append(l)
                    
        return sents, labels, active, passive    

    base = './data/style-ptb-annotated/'
    d = [ 'tense-voice', 'ppr-voice','tense-voice-ppr']
    splits = ['/train.tsv', '/test.tsv', '/valid.tsv']
    
    sents, labels, active, passive = [], [], [], []
    for d_ in d:
        for split in splits:
            path = base + d_ + split
            print("importing from {}".format(path))
            if(d_ == 'tense-voice'):
                s, l, a, p = preproc_voice(path, idx=2)
            elif(d_ == 'ppr-voice'):
                s, l, a, p = preproc_voice(path, idx=2)
            elif('tense-voice-ppr'):
                s, l, a, p = preproc_voice(path, idx=2)
            sents.extend(s)
            labels.extend(l)
            active.extend(a)
            passive.extend(p)

    assert(len(sents) == len(labels))
    labels, sents = shuffle_in_unison(labels, sents)    
    trainl, validl = split_dataset(sents)
    print("#sentences: {}".format(len(sents)))
    #annotated
    write_to_disk2(sents, labels, "./data/style-ptb-annotated/voices/voices.csv")
    write_to_disk2(sents[:trainl], labels[:trainl], "./data/style-ptb-annotated/voices/train.csv")
    write_to_disk2(sents[trainl:validl], labels[trainl:validl], "./data/style-ptb-annotated/voices/valid.csv")
    write_to_disk2(sents[validl:], labels[validl:], "./data/style-ptb-annotated/voices/test.csv")
    #non annotated
    write_to_disk(sents[:trainl], "./data/style-ptb/voices/train.txt")
    write_to_disk(sents[trainl:validl], "./data/style-ptb/voices/valid.txt")
    write_to_disk(sents[validl:], "./data/style-ptb/voices/test.txt")
    #label wise
    write_to_disk(list(set(active)), "./data/style-ptb/voices/active.txt")
    write_to_disk(list(set(passive)), "./data/style-ptb/voices/passive.txt")

def create_toy_dataset():
    word1 = "the"
    noun = {"0": ["boy", "man", "husband", "boyfriend", "waiter"], #gender
            "1": ["girl", "woman", "wife", "girlfriend", "waitress"]}
    word3 = "said"
    word4 = "the"
    subject = {"0": ["food", "meal", "dinner", "breakfast", "lunch", "pasta", "chicken"], #subject
                "1": ["service", "staff", "experience", "atmosphere","location","place"]}
    word6 = "is"
    adj = {"0": ["bad", "worst", "horrible", "spicy", "bland", "expensive","disgusting","mediocre"], #sentiment
            "1":["good", "great", "excellent", "decent", "amazing", "wonderful","reasonable"]}
    lines, x, y = [], [], []
    def create_cluster(word1, label1, word3, word4, label2, word6, label3):
        lines = []
        x = []
        y = []
        for m in noun[label1]:
            sent = []
            sent.append(m)
            for s in subject[label2]:
                sent.append(s)
                for a in adj[label3]:
                    sent.append(a)
                    l = [word1, sent[0], word3, word4, sent[1], word6, sent[2]]
                    x.append(l)
                    y.append(int(label1+label2+label3, 2))
                    lines.append(' '.join(l) + ' ,  ' + str(int(label1+label2+label3, 2)) + '\n')
                    sent.pop()
                sent.pop()
            sent.pop()
        return lines, x, y
        
    for n in noun:
        for s in subject:
            for a in adj:    
                lines_, x_, y_ = create_cluster(word1, n, word3, word4, s, word6, a) 
                lines.extend(lines_)
                x.extend(x_)
                y.extend(y_)

    print("Created toy dataset of size: {}".format(len(lines)))
    # random.shuffle(lines)
    with open('./data/yelp/toy.txt', 'w') as f:
        for line in lines:
            f.write(line)
    return lines, x, y

def create_data(dataset, num_classes=2, style='sentiment'): #convert raw data to csv file
    print("Creating {} dataset...".format(dataset))

    def load_sent_from_ds(path, label):
        sents = []
        with open(path) as f:
            for line in f:
                line = clean_line(line, [',']) # dont remmove anything else here because some lines will be blank
                line = [line.rstrip('\n')] + [' , '] + [str(label)] # :-2 is to remove \n
                sents.append(line)
        return sents
    
    def write_to_ds(path, sents):
        print("creating {}...".format(path))
        with open(path, 'w') as f:
            for s in sents:
                f.write(' '.join(s) + '\n')
    
    def create_split(split):
        data = []
        for lbl in range(num_classes):
            data.extend(load_sent_from_ds('./data/annotated/{}/{}.{}.{}'.format(dataset, style, split, str(lbl)), lbl))
        random.shuffle(data) 
        write_to_ds('./data/annotated/{}/{}.csv'.format(dataset, split), data)
    splits = ['train', 'test', 'valid']

    for split in splits:
        create_split(split)

def create_snli():
    def write_(path, sents):
        for lbl in sents:
            with open(path+lbl, 'w') as f:
                random.shuffle(sents[lbl])
                for line in sents[lbl]:
                    line = ' '.join(line).lower()
                    f.write(line + '\n')
        
    def load_sent_from_snli(path):
        sents = {}
        labels=['entailment','neutral','contradiction','-']
        with jsonlines.open(path, 'r') as f:
            for line in f:
                line['sentence1'] = clean_line(line['sentence1'])
                line['sentence2'] = clean_line(line['sentence2'])
                l1 = [word for word in line['sentence1'].split()]
                l2 = [word for word in line['sentence2'].split()]
                label = str(labels.index(line['gold_label']))
                if(label == '1' or label == '3'): #skip neutral and - labels
                    continue
                if(label == '2'):
                    label =  '1'
                l = l1 + ['.'] + l2
                if label not in sents:
                    sents[label] = []
                sents[label].append(l)
        return sents

    train = load_sent_from_snli('./data/annotated/snli/snli_1.0_train.jsonl')
    test = load_sent_from_snli('./data/annotated/snli/snli_1.0_test.jsonl')
    dev = load_sent_from_snli('./data/annotated/snli/snli_1.0_dev.jsonl')
    write_('./data/annotated/snli/nli.train.', train)
    write_('./data/annotated/snli/nli.test.', test)
    write_('./data/annotated/snli/nli.valid.', dev)

def create_dnli():
    
    print("Creating dnli")
    def write_(path, sents):
        for lbl in sents:
            with open(path+lbl, 'w') as f:
                random.shuffle(sents[lbl])
                for line in sents[lbl]:
                    line = ' '.join(line).lower()
                    f.write(line + '\n')
    def load_sent_from_dnli(path):
        sents = {}
        labels = ['positive', 'neutral', 'negative']
        with open(path, 'r') as f:
            data = json.load(f)
            for line in data:
                line['sentence1'] = clean_line(line['sentence1'])
                line['sentence2'] = clean_line(line['sentence2'])
                l1 = [word for word in line['sentence1'].split() if word !=
                      ',' and word != '.']
                l2 = [word for word in line['sentence2'].split() if word !=
                      ',' and word != '.']
                label = str(labels.index(line['label']))
                if(label == '1'):  # skip neutral
                    continue
                l = l1 + [' . '] + l2
                if label == '2':
                    label = '1'
                if label not in sents:
                    sents[label] = []
                sents[label].append(l)
        return sents
    train = load_sent_from_dnli('./data/annotated/dnli/train.json')
    test = load_sent_from_dnli('./data/annotated/dnli/test.json')
    valid = load_sent_from_dnli('./data/annotated/dnli/valid.json')
    write_('./data/annotated/dnli/nli.train.', train)
    write_('./data/annotated/dnli/nli.test.', test)
    write_('./data/annotated/dnli/nli.valid.', valid)

def create_scitail():
    print("Creating scitail")
    def write_(path, sents):
        for lbl in sents:
            with open(path+lbl, 'w') as f:
                random.shuffle(sents[lbl])
                for line in sents[lbl]:
                    line = ' '.join(line).lower()
                    f.write(line + '\n')
    def load_sent_from_scitail(path):
        sents = {}
        labels = ['entailment', 'neutral', '-']
        with jsonlines.open(path, 'r') as f:
            for line in f:
                line['sentence1'] = clean_line(line['sentence1'])
                line['sentence2'] = clean_line(line['sentence2'])
                l1 = [word for word in line['sentence1'].split() if word !=
                      ',' and word != '.']
                l2 = [word for word in line['sentence2'].split() if word !=
                      ',' and word != '.']
                label = str(labels.index(line['gold_label']))
                if(label == '2'):  # skip neutral
                    continue
                if(label == '2'):
                    label = '1'
                l = l1 + ['.'] + l2
                if label not in sents:
                    sents[label] = []
                sents[label].append(l)
        return sents
    train = load_sent_from_scitail('./data/annotated/scitail/train.txt')
    test = load_sent_from_scitail('./data/annotated/scitail/test.txt')
    valid = load_sent_from_scitail('./data/annotated/scitail/valid.txt')
    write_('./data/annotated/scitail/nli.train.', train)
    write_('./data/annotated/scitail/nli.test.', test)
    write_('./data/annotated/scitail/nli.valid.', valid)

def create_qqp():
    print("Creating qqp")
    def write_(path, sents):
        for lbl in sents:
            with open(path+lbl, 'w') as f:
                random.shuffle(sents[lbl])
                for line in sents[lbl]:
                    line = ' '.join(line).lower()
                    f.write(line + '\n')
    def load_sent_from_scitail(path):
        sents = {}
        labels = ['entailment', 'neutral', '-']
        with open(path, 'r') as f:
            for line in f:
                l, label = line.split(',')
                if label not in sents:
                    sents[label] = []
                sents[label].append(l.split())
        return sents
    train = load_sent_from_scitail('./data/annotated/qqp/train.csv')
    test = load_sent_from_scitail('./data/annotated/qqp/test.csv')
    valid = load_sent_from_scitail('./data/annotated/qqp/valid.csv')
    write_('./data/annotated/qqp/nli.train.', train)
    write_('./data/annotated/qqp/nli.test.', test)
    write_('./data/annotated/qqp/nli.valid.', valid)
    
if __name__ == "__main__":
    
    create_data("scitail", style="nli")
    # create_scitail()
    
    
    
