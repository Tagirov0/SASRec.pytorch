import os
import sys
import time
import copy
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue

from model import SASRec

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--predict_path', default=None, type=str)
parser.add_argument('--k_recs', default=10, type=int)

args = parser.parse_args()


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    preds = {}

    for u in tqdm(range(1, usernum)):
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if valid.get(u):
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = np.array(list(set(range(1, itemnum)).difference(rated)))
        np.random.shuffle(item_idx)

        scores = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        scores = scores[0].detach().cpu().numpy()

        unsorted_recs = scores.argpartition(-args.k_recs)[-args.k_recs:]
        unsorted_recs_score = scores[unsorted_recs]
        recs = unsorted_recs[(-unsorted_recs_score).argsort()]

        preds[u] = recs[:args.k_recs].tolist()

    return preds


if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
        
    if args.state_dict_path is not None:
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            
    
    if args.inference_only:
        model.eval()
        preds = evaluate(model, dataset, args)
        
        with open(f'sasrec_preds.json', 'w') as files:
            json.dump(preds, files)

    f.close()
    files.close()
    print("Done")
