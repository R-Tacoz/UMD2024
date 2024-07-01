# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch
import logging

# Set PATHs
PATH_TO_CUSTOMS = '../pkgs/'
PATH_TO_SENTEVAL = '../SentEval/'
PATH_TO_DATA = '../SentEval/data/'

sys.path.append(PATH_TO_CUSTOMS)
sys.path.insert(0, PATH_TO_SENTEVAL)

from transformers import BertModel, BertTokenizer
import senteval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

# SentEval prepare and batcher
def prepare(params, samples):
    params.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    params.bert.eval()
    params.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return

def batcher(params, batch):
    batch = [" ".join(sent) if sent != [] else '.' for sent in batch]
    
    out = params.tokenizer(batch, max_length=128, truncation=True, padding=True, return_tensors='pt').to(device)
    out = params.bert(**out).last_hidden_state.squeeze()
    out = torch.mean(out, 1)
    
    
    # print('out.shape: ', out.shape)
    
    embeddings = out.detach().to(device_cpu)
    
    return embeddings


# Set params for SentEval
params_senteval = {
    'task_path': PATH_TO_DATA, 
    'usepytorch': True, 
    'kfold': 5,
    'classifier': {
        'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 
        'epoch_size': 2},
}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICKRelatedness', 'STSBenchmark']
                    #   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                    #   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                    #   'Length', 'WordContent', 'Depth', 'TopConstituents',
                    #   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                    #   'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
