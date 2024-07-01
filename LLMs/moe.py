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
import argparse

# Set PATHs
PATH_TO_CUSTOMS = '../pkgs/'
PATH_TO_SENTEVAL = '../SentEval/'
PATH_TO_DATA = '../SentEval/data/'

sys.path.append(PATH_TO_CUSTOMS)
sys.path.insert(0, PATH_TO_SENTEVAL)

import senteval

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel, PhiForCausalLM, Phi3ForCausalLM  # noqa: F402
from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration, T5Tokenizer
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import BitsAndBytesConfig

from peft import PeftModel

PATH_TO_DATA = './data'

device_cpu = torch.device('cpu')

def load_model(base_model, peft) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    if 'neo' in base_model:
        tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    elif 'phi-2' in base_model:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
            use_fast=False, # needed for now, should be fixed soon
        )
        # tokenizer.pad_token = tokenizer.eos_token
    elif 'Phi-3' in base_model:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
            use_fast=False, # needed for now, should be fixed soon
        )
        # tokenizer.pad_token = tokenizer.eos_token
    elif 'gemma' in base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    elif 'OLMo' in base_model:
        tokenizer = OLMoTokenizerFast.from_pretrained(base_model, revision="step20000-tokens84B")
    elif 'mis' in base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "right"
        tokenizer.pad_token_id = 0 
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )


    if 'llama' in base_model:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )
    elif 'phi-2' in base_model:
        print('AutoModelForCausalLM.from_pretrained:', AutoModelForCausalLM.from_pretrained)
        model = PhiForCausalLM.from_pretrained(
            base_model, 
            trust_remote_code=True, 
            torch_dtype="auto", 
            # load_in_8bit=True,
        )
    elif 'Phi-3' in base_model:
        # print('AutoModelForCausalLM.from_pretrained:', AutoModelForCausalLM.from_pretrained)
        model = Phi3ForCausalLM.from_pretrained(
            base_model, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
    elif 'neo' in base_model:
        model = GPTNeoForCausalLM.from_pretrained(
            base_model,
        )
    elif 'gemma' in base_model:
        model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    elif 'OLMo' in base_model:
        from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
        model = OLMoForCausalLM.from_pretrained(base_model)
    elif 'mis' in base_model:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            quantization_config=nf4_config,
            use_cache=False
        )    
        
    if peft is not None:
        model = PeftModel.from_pretrained(
                model, peft
            )

    model.eval()

    return model, tokenizer


# SentEval prepare and batcher
def prepare(params, samples):
    params.model, params.tokenizer = load_model(base_model, peft)
    return

def batcher(params, batch):
    
    with torch.no_grad():
        batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
        
        # template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
        # inputs = params.tokenizer([template.replace('*sent_0*', i).replace('_', ' ') for i in batch], padding=True, return_tensors="pt")
        inputs = params.tokenizer(batch, padding=True, return_tensors="pt")
        res = params.model(inputs["input_ids"].cuda(), output_hidden_states=True, return_dict=True)
        lst_token_hidden_state = res.hidden_states[-1][:, -1, :].detach().to(device_cpu)
        
        if peft is not None:
            selected_adapter_idxs = res.selected_adapter_idxs.permute(2, 0, 1, 3)[1:]
            weighted_matrix = res.weighted_matrix
            weighted_matrix = weighted_matrix.reshape(weighted_matrix.shape[0], -1).detach().to(device_cpu)
        
            # embeddings = torch.cat([adapter_idxs, weighted_matrix], dim=-1)
            merged_adapter = params.model.base_model.model.adapter_zoo.composed_ins_adapter
            
            merged_adapter = torch.cat([torch.cat([item.unsqueeze(1) for item in merged_adapter[i] if item is not None], dim=1).unsqueeze(1) for i in range(32)], dim=1).mean(1).mean(1).detach().to(device_cpu)
            last_hidden_merge_adapter = torch.cat([last_hidden_state, merged_adapter], dim=-1)
        
        # embeddings = last_hidden_state
        # embeddings = last_hidden_merge_adapter
        embeddings = lst_token_hidden_state
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="", required=True)
    parser.add_argument('--peft', default=None, required=False)
    parser.add_argument('--max_input_length', type=int, default=8)
    args = parser.parse_args()
    base_model = args.base_model
    peft = args.peft
    
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICKRelatedness', 'STSBenchmark']
                    #   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                    #   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                    #   'Length', 'WordContent', 'Depth', 'TopConstituents',
                    #   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                    #   'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
