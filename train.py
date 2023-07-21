import argparse
import time
import os
import random
import collections
import torch
from model import   LSTMClassifier, DiversityLSTMClassifier, AAE, VanillaAE
from vocab import Vocab
from utils import *
from batchify import get_batches_annotated, get_batches_mt

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument('--save-dir', default='checkpoints-supervised', required=True, metavar='DIR',
                    help='directory to save checkpoints and outputs')
parser.add_argument('--load-model', default='', metavar='FILE',
                    help='path to load checkpoint if specified')
# Architecture arguments
parser.add_argument('--vocab-size', type=int, default=25000, metavar='N',
                    help='keep N most frequent words in vocabulary')
parser.add_argument('--dim-emb', type=int, default=300, metavar='D',
                    help='dimension of word embedding')
parser.add_argument('--dim-z', type=int, default=128, metavar='D',
                    help='dimension of latent variable z')
parser.add_argument('--dim-h', type=int, default=512, metavar='D',
                    help='dimension of hidden state per layer')
parser.add_argument('--dim-d', type=int, default=512, metavar='D',
                    help='dimension of hidden state in AAE discriminator')
parser.add_argument('--nlayers', type=int, default=1, metavar='N',
                    help='number of layers')   
parser.add_argument('--model-type', default='lstmclassifier', metavar='M',
                    choices=['van','aae','lstmclassifier','ortholstmclassifier','divlstmclassifier'],
                    help='which model to learn')                 

# Training arguments
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size')
# Others
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--output-size', type=int, default=2, metavar='N',
                    help='number of classes in dataset')
parser.add_argument('--train', metavar='FILE', required=True,
                    help='path to training file')
parser.add_argument('--valid', metavar='FILE', required=True,
                    help='path to validation file')
parser.add_argument('--dropout', type=float, default=0.3, metavar='DROP',
                    help='dropout probability (0 = no dropout)')
parser.add_argument('--glove', '-g', action='store_true')
parser.add_argument('--gpu', type=int, default=0, metavar='N',
                    help='ID of gpu')
parser.add_argument('--token-noise', default='0,0,0,0', metavar='P,P,P,K',
                    help='word drop prob, blank prob, substitute prob'
                         'max word shuffle distance')
parser.add_argument('--token-drop-attr', '-tka', action='store_true', help='drop tokens acc to attribution scores')
parser.add_argument('--zeta', type=float, default=0.0, metavar='R',
                    help='weight for embedding noise')
parser.add_argument('--noise-type', default='none', metavar='M',
                    choices=['hollow', 'uniform', 'shifted-gau', 'centered-gau', 'attribution'], 
                    help='model for the noise hypersphere')
parser.add_argument('--lambda-kl', type=float, default=0, metavar='R',
                    help='weight for kl term in VAE')
parser.add_argument('--lambda-adv', type=float, default=0, metavar='R',
                    help='weight for adversarial loss in AAE')
parser.add_argument('--lambda-p', type=float, default=0, metavar='R',
                    help='weight for L1 penalty on posterior log-variance')
parser.add_argument('--lambda-c', type=float, default=0, metavar='R',
                    help='weight for conicity penalty on hidden states of RNN')
parser.add_argument('--lambda-t', type=float, default=0, metavar='R',
                    help='weight for triplet loss')
parser.add_argument('--fn', default=None, metavar='M',
                    choices=['logistic', 'sigmoid', 'linear', 'tanh'], 
                    help='model for the zeta anneal fn')
parser.add_argument('--dataset', default='default', help='to specify the dataset we are training on (needed for AGAAE)')
parser.add_argument('--save-all', action='store_true',default=False, help='save model at every epoch')


def evaluate(model, batches):
    model.eval()
    meters = collections.defaultdict(lambda: AverageMeter())
    with torch.no_grad():
        for inputs, targets, lbls in batches:
            losses = model.autoenc(inputs=inputs, args=args, targets=targets, lbls=lbls, is_train=False)
            if(isinstance(model, LSTMClassifier) or isinstance(model, DiversityLSTMClassifier)):
                _, _, outs = model(inputs, args=args, is_train=False)
                batch_acc = compute_acc(outs, lbls)
                meters['acc'].update(batch_acc)
            for k, v in losses.items():
                if v is not None:
                    meters[k].update(v.item(), inputs.size(1))
            
    loss = model.loss({k: meter.avg for k, meter in meters.items()})
    meters['loss'].update(loss)
    return meters


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log_file = os.path.join(args.save_dir, 'log.txt')
    logging(str(args), log_file)

    # Prepare data
    train_sents = load_sent_mt(args.train)
    logging('# train sents {}, tokens {}'.format(len(train_sents), sum(len(s) for s in train_sents)), log_file)
    valid_sents = load_sent_mt(args.valid)
    logging('# valid sents {}, tokens {}'.format(len(valid_sents), sum(len(s) for s in valid_sents)), log_file)
    vocab_file = os.path.join(args.save_dir, 'vocab.txt')
    if not os.path.isfile(vocab_file):
        print("Creating vocab...")
        Vocab.build(train_sents[1], vocab_file, args.vocab_size)
        print("Done creating vocab!")
    vocab = Vocab(vocab_file)
    logging('# vocab size {}'.format(vocab.size), log_file)

    set_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.gpu)) if cuda else 'cpu'
    model = {'van':VanillaAE,'aae':AAE, 'lstmclassifier': LSTMClassifier, 'divlstmclassifier':DiversityLSTMClassifier}[args.model_type](
        vocab, args).to(device)
    if args.load_model:
        print("Loading model from mem...")
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['model'])
        model.flatten()
        print("Done loading model!")
    logging('# model parameters: {}'.format(
        sum(x.data.nelement() for x in model.parameters())), log_file)


    train_batches, _ = get_batches_mt(train_sents, vocab, args.batch_size, device, num_classes=args.output_size)
    valid_batches, _ = get_batches_mt(valid_sents, vocab, args.batch_size, device, num_classes=args.output_size)
    best_val_loss = None

    for epoch in range(args.epochs):
        start_time = time.time()
        logging('-' * 80, log_file)
        model.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(train_batches)))   
        random.shuffle(indices)
        
        for i, idx in enumerate(indices):
            inputs, targets, lbls = train_batches[idx]
            losses = model.autoenc(inputs=inputs, args=args, targets=targets, lbls=lbls, is_train=True)
            losses['loss'] = model.loss(losses)
            model.step(losses)
        
            for k, v in losses.items():
                meters[k].update(v.item())

            if (i + 1) % args.log_interval == 0:
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(
                    epoch + 1, i + 1, len(indices))
                for k, meter in meters.items():
                    log_output += ' {} {:.2f},'.format(k, meter.avg)
                    meter.clear()
                logging(log_output, log_file)
                
        valid_meters = evaluate(model, valid_batches)
        logging('-' * 80, log_file)
        log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
            epoch + 1, time.time() - start_time)
        #print sample recon
        # print(model.vocab.get_sent(inputs.t().tolist()))
        # print(model.vocab.get_sent(targets.t().tolist()))
        for k, meter in valid_meters.items():
            log_output += ' {} {:.2f},'.format(k, meter.avg)
        if not best_val_loss or valid_meters['loss'].avg < best_val_loss:
            log_output += ' | saving model'
            ckpt = {'args': args, 'model': model.state_dict()}
            if(not os.path.exists(os.path.join(args.save_dir, 'checkpoints'))):
                os.mkdir(os.path.join(args.save_dir, 'checkpoints'))
            torch.save(ckpt, os.path.join(args.save_dir, 'checkpoints','model_best.pt'))
            best_val_loss = valid_meters['loss'].avg
        if args.save_all:
            ckpt = {'args': args, 'model': model.state_dict()}
            torch.save(ckpt, os.path.join(args.save_dir, 'checkpoints','model_best.pt{}'.format(epoch)))
        logging(log_output, log_file)
    logging('Done training {}'.format(args.save_dir), log_file)


if __name__ == '__main__':
    args = parser.parse_args()
    args.token_noise = [float(x) for x in args.token_noise.split(',')]
    main(args)
