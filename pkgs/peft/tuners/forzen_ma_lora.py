import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftType, transpose, ModulesToSaveWrapper
from ..config import PeftConfig

from copy import deepcopy, copy

import ot
import os
from huggingface_hub import hf_hub_download
import sys
sys.path.append('/nfshomes/litzy/mixture-of-adapters/')
sys.path.append('/home/litzy619/mixture-of-adapters/')
from ot_pytorch import sink, sink_stabilized

from .lora import onload_layer

from dataclasses import dataclass, field

def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None

if is_bnb_available():
    import bitsandbytes as bnb
    
# Define a function for calculating cosine similarity with gradient checkpointing
def cosine_sim_with_checkpointing(emb, av):
    return F.cosine_similarity(emb, av, dim=2).mean(1).unsqueeze(1)

# Define a function for computing bmm with gradient checkpointing
def bmm_with_checkpointing(a, inputs_embeds):
    return torch.bmm(a.permute(0, 2, 1), inputs_embeds)


WEIGHTS_NAME = "adapter_model.bin"
WEIGHTS_NAME2 = "adapter_model.safetensors"


@dataclass
class Forzen_MA_LoraConfig(PeftConfig):
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    number_of_adapter_pre_layer: int = field(default=8, metadata={"help": "The number of A-B pairs"})
    dynamic_adapter_pool: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    selective_num: int = field(default=1, metadata={"help": "The number of A-B pairs chosen for recomposing"})
    pool_selective_training: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    pool_selective_inference: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    ot_diversified_dispatcher: bool = field(
        default=False, metadata={"help": "whether do use ot to improve the diversity of assignment"}
    )
    
    adaptive_ratio: float = field(
        default=1.0, metadata={"help": "whether do use ot to improve the diversity of assignment"}
    )
    adaptive_ratio_decay: float = field(
        default=1.0, metadata={"help": "whether do use ot to improve the diversity of assignment"}
    )
    random_ratio: float = field(
        default=1.0, metadata={"help": "whether do use ot to improve the diversity of assignment"}
    )
    random_ratio_decay: float = field(
        default=0.995, metadata={"help": "whether do use ot to improve the diversity of assignment"}
    )
    scaling_ratio: float = field(
        default=1.0, metadata={"help": "whether do use ot to improve the diversity of assignment"}
    )
    scaling_ratio_decay: float = field(
        default=0.995, metadata={"help": "whether do use ot to improve the diversity of assignment"}
    )
    input_based_adapter_selection: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    simple_instance_matching: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    simple_hidden_matching: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    layer_to_lora: list = field(default_factory=list)
    allow_empty_lora: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    insert_zero_lora: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    pretrain_adapter_list: str = field(default='', metadata={"help": "whether do selective inference"}
    )
    allow_negative_weights: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    train_lm_head: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    lora_lm_head: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    adapter_emb_size: int = field(default=512, metadata={"help": "The number of A-B pairs chosen for recomposing"})
    
    # discard
    random_routing: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    random_routing_inference: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )

    def __post_init__(self):
        self.peft_type = PeftType.FORZEN_MA_LORA


class Adapter_Zoo(torch.nn.Module):

    def __init__(self, num_layer, emb_size, config):
        super().__init__()
        
        self.adapter_emb_size = config.adapter_emb_size
        self.emb_size = emb_size
        try:
            self.num_layer = config.num_layers
        except:
            if 'OLM' in config.base_model_name_or_path:
                self.num_layer = 16
            elif 'gemma' in config.base_model_name_or_path:
                self.num_layer = 18
            else:
                self.num_layer = 32

        self.config = config
        self.target_modules = config.target_modules
        self.num_adapter_each_layer = config.number_of_adapter_pre_layer
        self.input_based_adapter_selection = config.input_based_adapter_selection
        self.simple_instance_matching = config.simple_instance_matching
        self.simple_hidden_matching = config.simple_hidden_matching
        self.allow_negative_weights = config.allow_negative_weights
        
        # self.lora_proj = nn.Linear(self.emb_size, self.adapter_emb_size)
        self.lora_proj = nn.ModuleList()
        # self.lora_proj.append(nn.Linear(self.emb_size, config.r))
        # self.lora_proj.append(nn.Linear(config.r, self.adapter_emb_size))
        self.lora_proj.append(nn.Linear(self.adapter_emb_size, config.r))
        self.lora_proj.append(nn.Linear(config.r, self.emb_size))
        nn.init.kaiming_uniform_(self.lora_proj[0].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_proj[1].weight, a=math.sqrt(5))
        # nn.init.zeros_(self.lora_proj[1].weight)
        
        
    def load_trained_adapters(self, AB_group):
        
        print('AB_group: ', AB_group)
        
        module_groups = [AB_group[key] for key in AB_group]
        keys = list(AB_group.keys())
        
        self.module_dict = {}
        for idx, modules in enumerate(module_groups):
            for module in modules:
                self.module_dict[module] = idx
            
        print('self.module_dict:', self.module_dict)
        
        # load pretrained models
        pretrained_adapters=[]
        with open(self.config.pretrain_adapter_list,'r') as f:
            for line in f:
                pretrained_adapters.append(line.strip('\n'))
                
        print('pretrained_adapters: ', pretrained_adapters)
        
        self.lora_A_params_dict : Dict[str, OrderedDict[str, torch.Tensor]] = {
            g_id:{l_id:[] for l_id in range(self.num_layer)} for g_id in range(len(module_groups))
        }
        self.lora_B_params_dict : Dict[str, OrderedDict[str, torch.Tensor]] = {
            g_id:{l_id:[] for l_id in range(self.num_layer)} for g_id in range(len(module_groups))
        }
        
        
        for model_id in pretrained_adapters:
            use_safetensors = False
            # load weights if any
            if os.path.exists(os.path.join(model_id, WEIGHTS_NAME)):
                filename = os.path.join(model_id, WEIGHTS_NAME)
            elif os.path.exists(os.path.join(model_id, WEIGHTS_NAME2)):
                filename = os.path.join(model_id, WEIGHTS_NAME2)
                use_safetensors = True
            else:
                try:
                    filename = hf_hub_download(model_id, WEIGHTS_NAME)
                except:  # noqa
                    try:
                        filename = hf_hub_download(model_id, WEIGHTS_NAME2)
                        use_safetensors = True
                    except:
                        raise ValueError(
                            f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                            f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
                        )

            if use_safetensors:
                from safetensors.torch import load_file as safe_load_file
                adapters_weights = safe_load_file(filename, device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                adapters_weights = torch.load(
                    filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                
            for name, param in adapters_weights.items():
                # print(f"Name: {name}, Shape: {param.shape}")
                layer = int(name.split('.')[-5])
                module_type = name.split('.')[-3]
                lora_type = name.split('.')[-2]
                gid = self.module_dict[module_type]
                if lora_type == 'lora_A':
                    self.lora_A_params_dict[gid][layer].append(param)
                elif lora_type == 'lora_B':
                    self.lora_B_params_dict[gid][layer].append(param)  
                    
        
        all_avail = 0
        self.num_avail_adapters = [0 for gid in self.lora_A_params_dict]    
        self.cum_num_avail_adapters = []   
        self.adapter_splits = []      
        for gid in self.lora_A_params_dict:
            
            self.adapter_splits.append([])
            for lid in range(self.num_layer):
                self.adapter_splits[gid].append(self.num_avail_adapters[gid] + len(self.lora_A_params_dict[gid][lid]))
                self.num_avail_adapters[gid] += len(self.lora_A_params_dict[gid][lid])
                all_avail += len(self.lora_A_params_dict[gid][lid])
           
        print('num_avail_adapters: ', self.num_avail_adapters)
        print('self.adapter_splits: ', self.adapter_splits) 
        print('all_avail: ', all_avail)
        
        sum = 0
        for gid in self.lora_A_params_dict:
            self.cum_num_avail_adapters.append(sum + self.num_avail_adapters[gid])
            sum += self.num_avail_adapters[gid]
        
        if hasattr(self.config, "allow_empty_lora") and self.config.allow_empty_lora:
            self.lora_backbone_emb = torch.nn.Parameter(torch.ones((self.num_layer, 1, 1, self.adapter_emb_size)), requires_grad=True)
            nn.init.xavier_uniform_(self.lora_backbone_emb)
            self.allow_empty_lora = True
        else:
            self.allow_empty_lora = False
        
        # # Create a parameter tensor
        self.lora_adapter_emb = torch.nn.Parameter(torch.ones((all_avail, self.adapter_emb_size)), requires_grad=True)
        nn.init.xavier_uniform_(self.lora_adapter_emb)

        if 'lm_head' in self.target_modules:
            self.weighted_matrix = [[None for _ in range(len(self.target_modules)-1)] for _ in range(self.num_layer)]
        else:
            self.weighted_matrix = [[None for _ in range(len(self.target_modules))] for _ in range(self.num_layer)]
           
        self.composed_ins_adapter = [[None for _ in range(len(self.target_modules))] for _ in range(self.num_layer)]
         
        self.m = nn.Softmax(dim=-1)
        
        self.task_head_list = [14336]
        self.lora_specific_head = nn.ModuleList()
        for i in range(len(self.task_head_list)):
            self.lora_specific_head.append(nn.ModuleList())
            # self.lora_specific_head[i].append(nn.Linear(self.task_head_list[i],self.config.r))
            # self.lora_specific_head[i].append(nn.Linear(self.config.r,self.adapter_emb_size))
            
            self.lora_specific_head[i].append(nn.Linear(self.adapter_emb_size,self.config.r))
            self.lora_specific_head[i].append(nn.Linear(self.config.r,self.task_head_list[i]))
       
            nn.init.kaiming_uniform_(self.lora_specific_head[i][0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_specific_head[i][1].weight)
            
        # nn.init.xavier_uniform_(self.lora_specific_head)
        
            
    def select_best_for_input(self, input, sel_num, layer_idx = None, module_name = None, ratio = 1.0, random_ratio=1.0, inputs_embeds = None):

        inputs_embeds = inputs_embeds.float()
        
        # print('ori inputs_embeds.shape: ', inputs_embeds.shape)

        for token_idx in range(0,inputs_embeds.shape[1]):
            same = inputs_embeds[0,token_idx]
            if all([torch.equal(same, inputs_embeds[i_idx,token_idx]) for i_idx in range(1,inputs_embeds.shape[0])]):
                continue
            else:
                break
            
        zeros = torch.zeros(inputs_embeds.shape[-1]).to(inputs_embeds.device)
        for last_token_idx in range(inputs_embeds.shape[1]-1,-1,-1):
            if any([torch.equal(inputs_embeds[i_idx,last_token_idx],zeros) for i_idx in range(inputs_embeds.shape[0])]):
                continue
            else:
                break
            
        # print(f'token_idx {token_idx} last_token_idx {last_token_idx}')
        inputs_embeds = inputs_embeds[:, token_idx:last_token_idx+1, :]
        # print('after inputs_embeds.shape: ', inputs_embeds.shape)
        
        # for instance_idx in range(inputs_embeds.shape[0]):
        #     print(f'instance i {inputs_embeds[instance_idx]}')
        
        if self.allow_empty_lora:
            backbone_emb = self.lora_backbone_emb[layer_idx,0,:]
        
        gid = self.module_dict[module_name]

        if ratio < 1.0:
            start_idx = max(0, int(layer_idx - ratio * self.num_layer))
            end_idx = min(self.num_layer - 1, int(layer_idx + ratio * self.num_layer + 1))
        else:
            start_idx = 0
            end_idx = self.num_layer - 1
            
        location_helper = []
        
        for l_id in range(start_idx, end_idx+1):
            location_helper.append(len(self.lora_A_params_dict[gid][l_id]))
        
        if gid == 0:
            lora_adapter_emb = self.lora_adapter_emb[:self.num_avail_adapters[gid]]
        else:
            lora_adapter_emb = self.lora_adapter_emb[self.cum_num_avail_adapters[gid-1]:self.cum_num_avail_adapters[gid]]
        
        if start_idx == 0:
            ori_emb = lora_adapter_emb[0:self.adapter_splits[gid][end_idx]].to(input.device)
        else:
            ori_emb = lora_adapter_emb[self.adapter_splits[gid][start_idx-1]:self.adapter_splits[gid][end_idx]].to(input.device)
            
        if self.allow_empty_lora:
            reshaped_emb = torch.cat([backbone_emb.to(input.device), ori_emb], dim=0)
        else:
            reshaped_emb = ori_emb
            
        # print('reshaped_emb.shape: ', reshaped_emb.shape)
        
        if input.shape[-1] != self.emb_size and self.simple_hidden_matching:
            self.lora_specific_head[self.task_head_list.index(input.shape[-1])] = self.lora_specific_head[self.task_head_list.index(input.shape[-1])].to(input.device)
            for module in self.lora_specific_head[self.task_head_list.index(input.shape[-1])]:
                reshaped_emb = module(reshaped_emb.float())
        elif inputs_embeds.shape[-1] != self.emb_size and not self.simple_hidden_matching:
            self.lora_specific_head[self.task_head_list.index(inputs_embeds.shape[-1])] = self.lora_specific_head[self.task_head_list.index(inputs_embeds.shape[-1])].to(inputs_embeds.device)
            for module in self.lora_specific_head[self.task_head_list.index(inputs_embeds.shape[-1])]:
                reshaped_emb = module(reshaped_emb.float())
        else:
            for module in self.lora_proj:
                # input = module(input.float())
                reshaped_emb = module(reshaped_emb.float())
        
        if self.input_based_adapter_selection:
            if self.simple_hidden_matching:
                if self.allow_negative_weights:
                    all_inner_prod = torch.matmul(input.float(), reshaped_emb.permute(1, 0)).mean(1).unsqueeze(1)
                else:
                    all_inner_prod = self.m(torch.matmul(input.float(), reshaped_emb.permute(1, 0))).mean(1).unsqueeze(1)
            else:
                if self.allow_negative_weights:
                    all_inner_prod = torch.matmul(inputs_embeds.float(), reshaped_emb.permute(1, 0)).mean(1).unsqueeze(1)
                else:
                    all_inner_prod = self.m(torch.matmul(inputs_embeds.float(), reshaped_emb.permute(1, 0))).mean(1).unsqueeze(1)
        else:
            if self.allow_negative_weights:
                all_inner_prod = torch.matmul(input.float(), reshaped_emb.permute(1, 0))
            else:
                all_inner_prod = self.m(torch.matmul(input.float(), reshaped_emb.permute(1, 0)))
        
        _, topk_indices = torch.topk(all_inner_prod.squeeze(1)[:,1:], min(sel_num, all_inner_prod.shape[-1]), dim=-1)
        
        if self.allow_empty_lora:
            routing_weights, selected_experts = torch.topk(all_inner_prod.sum(1, keepdim=False)[:,1:], min(sel_num, all_inner_prod.shape[-1]), dim=-1)
        else:
            routing_weights, selected_experts = torch.topk(all_inner_prod.sum(1, keepdim=False), min(sel_num, all_inner_prod.shape[-1]), dim=-1)
            
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        used_experts = torch.unique(selected_experts.reshape(-1))
        
        
        # print('used_experts:', used_experts)
        
        # print('selected_experts:', selected_experts)
        
        # print('routing_weights.shape: ', routing_weights.shape)
        
        
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=all_inner_prod.shape[-1]).permute(2, 1, 0)
        
        # print('expert_mask.shape: ', expert_mask.shape)
        # print(expert_mask)
        
        # for expert_idx in range(all_inner_prod.shape[-1]):
        #     idx, top_x = torch.where(expert_mask[expert_idx])
        #     print(f'expert_idx {expert_idx} idx {idx} top_x {top_x}')
        #     print(routing_weights[top_x, idx, None])
    
        A_param = {}
        B_param = {}
        
        for i, idx in enumerate(used_experts):
            # if (i/sel_num) <= random_ratio and self.training and sel_num != 1:
            #     lid = rand_idx[i] // location_helper[0]
            #     act_idx = rand_idx[i] % location_helper[0]
            #     A_param.append(self.lora_A_params_dict[gid][lid][act_idx])
            #     B_param.append(self.lora_B_params_dict[gid][lid][act_idx])
            #     A_param[idx] = self.lora_A_params_dict[gid][lid][act_idx]
            #     B_param[idx] = self.lora_B_params_dict[gid][lid][act_idx]
            # else:
            lid = idx.item() // location_helper[0]
            act_idx = idx.item() % location_helper[0]
            # A_param.append(self.lora_A_params_dict[gid][lid][act_idx])
            # B_param.append(self.lora_B_params_dict[gid][lid][act_idx])
            A_param[idx.item()] = self.lora_A_params_dict[gid][lid][act_idx]
            B_param[idx.item()] = self.lora_B_params_dict[gid][lid][act_idx]
            
        self.weighted_matrix[layer_idx][self.target_modules.index(module_name)] = routing_weights

        # composed_ins_adapter = []
        # for ins_idx in range(topk_indices.shape[0]):
        #     merged_one = [ori_emb[idx:idx+1]*all_inner_prod[ins_idx,:,idx+1] for idx in topk_indices[ins_idx]]
        #     merged_one = torch.cat(merged_one, dim=0).mean(0).unsqueeze(0)
        #     composed_ins_adapter.append(merged_one)
            
        # self.composed_ins_adapter[layer_idx][self.target_modules.index(module_name)] = torch.cat(composed_ins_adapter, dim=0)
        
        # topk_indices = torch.cat([topk_idxs.unsqueeze(0), topk_indices], dim=0)
        
        return topk_indices, A_param, B_param, routing_weights, used_experts, expert_mask
        

class Forzen_MA_LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """
    
    prefix: str = "lora_"

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model

        if 'llama' in self.peft_config.base_model_name_or_path or 'mis' in self.peft_config.base_model_name_or_path:
            emb_size = 4096
        elif 'phi' in self.peft_config.base_model_name_or_path:
            emb_size = 2560
        elif 'gemma' in self.peft_config.base_model_name_or_path:
            emb_size = 2048
        elif 'Phi' in self.peft_config.base_model_name_or_path:
            emb_size = 3072
        else:
            emb_size = 2048

        self.model.adapter_zoo = Adapter_Zoo(num_layer=config.num_layers if hasattr(config, 'num_layers') else 32, emb_size=emb_size, config = self.peft_config)
        self._find_and_replace()
        
        self.model.adapter_zoo.load_trained_adapters(self.AB_group)  
        mark_only_lora_as_trainable(self.model, self.peft_config.bias, self.peft_config.insert_zero_lora, config.train_lm_head, config.lora_lm_head)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        # quantization_config = getattr(self.model, "quantization_config", False)
        
        if (loaded_in_8bit and not is_bnb_available()):
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")

        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
            "number_of_adapter_pre_layer": self.peft_config.number_of_adapter_pre_layer,
        }

        key_list = [key for key, _ in self.model.named_modules()]
        
        # self.A_group = {}
        # self.B_group = {}
        self.AB_group = {}
        
        print('self.peft_config.target_modules:', self.peft_config.target_modules)
        
        for key in key_list:
            if hasattr(self.peft_config, "layer_to_lora") and len(self.peft_config.layer_to_lora) > 0:
                if not any(['.' + str(item) + '.' in key for item in self.peft_config.layer_to_lora]):
                    continue
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                
                # print('target_name: ', target_name)
                if isinstance(target, bnb.nn.Linear4bit):
                    # print('choice 4bit')
                    if self.peft_config.enable_lora is None:
                        new_module = Linear4bit(target.in_features, target.out_features, bias=bias, layer=int(key.split('.')[2]), adapter_zoo=self.model.adapter_zoo, proj=key.split('.')[-1], peft_config = self.peft_config, **kwargs)
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = Linear4bit(target.in_features, target.out_features, bias=bias, layer=int(key.split('.')[2] if not 'OLM' in self.peft_config.base_model_name_or_path else key.split('.')[-2]), adapter_zoo=self.model.adapter_zoo, proj=key.split('.')[-1], peft_config = self.peft_config, **kwargs)
                        
                    key_m = tuple([target.in_features, target.out_features])
                    
                    if key_m not in self.AB_group:
                        self.AB_group[key_m] = [key.split('.')[-1]]
                    else:
                        if key.split('.')[-1] not in self.AB_group[key_m]:
                            self.AB_group[key_m].append(key.split('.')[-1])
                    
                elif loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    print('choice 1')
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, layer=int(key.split('.')[2]), adapter_zoo=self.model.adapter_zoo, proj=key.split('.')[-1], peft_config = self.peft_config, **kwargs)
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    # currently used
                    if 'lm_head' == target_name and self.peft_config.lora_lm_head:
                        new_module = Pure_Linear(target.in_features, target.out_features, bias=bias, peft_config = self.peft_config, **kwargs)
                    else:
                        new_module = Linear(target.in_features, target.out_features, bias=bias, layer=int(key.split('.')[2] if not 'OLM' in self.peft_config.base_model_name_or_path else key.split('.')[-2]), adapter_zoo=self.model.adapter_zoo, proj=key.split('.')[-1], peft_config = self.peft_config, **kwargs)

                        key_m = tuple([target.in_features, target.out_features])
                        
                        if key_m not in self.AB_group:
                            self.AB_group[key_m] = [key.split('.')[-1]]
                        else:
                            if key.split('.')[-1] not in self.AB_group[key_m]:
                                self.AB_group[key_m].append(key.split('.')[-1])
                    
                elif self.peft_config.enable_lora is not None:
                    kwargs.update({"enable_lora": self.peft_config.enable_lora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = self.peft_config.fan_in_fan_out = False
                    new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )
            
        print('self.AB_group:')
        print(self.AB_group)

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)
       
    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names = None,
    ):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge LORA layers when the model is gptq quantized")

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = self._get_submodules(key)
            except AttributeError:
                continue
            
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        # check if the module is itself a tuner layer
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)

        return self.model 
        
    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names = None
    ) -> torch.nn.Module:
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )
        
    


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none", insert_zero_lora: bool = False, train_lm_head: bool = False, lora_lm_head: bool = False) -> None:
    for n, p in model.named_parameters():
        # print('n: ', n)
        # if "lora_" not in n:
        #     p.requires_grad = False
        # elif insert_zero_lora and ('lora_A.0.' in n or 'lora_B.0.' in n):
        #     p.requires_grad = False
        # else:
        #     print('***trainable param: ', n)
        # if ('lora_A.' in n or 'lora_B.' in n):
        #     p.requires_grad = False
        # elif "lora_" not in n:
        #     p.requires_grad = False
        # elif "lora_scaling" in n:
        #     p.requires_grad = False
        if 'adapter_zoo.' not in n:
            p.requires_grad = False
            
        if lora_lm_head:
            if 'lm_head' in n and 'lora' in n and train_lm_head:
                p.requires_grad = True
        else:
            if 'lm_head' in n and train_lm_head:
                p.requires_grad = True
            
        
    print('bias:', bias)
    
    print('all trainable params:')
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)
            
    print('---end---')
    
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError
    

    
    


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
        number_of_adapter_pre_layer: int = 1,
        layer: int = 1,
        proj: str = '',
        peft_config = None
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False
        self.number_of_adapter_pre_layer = number_of_adapter_pre_layer
        self.layer = layer
        self.proj = proj
        self.peft_config = peft_config


def L1_reg(G):
    return np.sum(G)


def d_L1_reg(G):
    return np.ones_like(G)


class Pure_Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        number_of_adapter_pre_layer: int = 1,
        layer: int = 1,
        proj: str = '',
        adapter_zoo = None,
        peft_config = None,
        **kwargs,
    ):
        # print('in_features: ', in_features)
        # print('out_features: ', out_features)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights, number_of_adapter_pre_layer = number_of_adapter_pre_layer, layer = layer, proj = proj, peft_config = peft_config)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = False
        # print('self.merged:', self.merged)

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype

        if self.disable_adapters:
            if self.r > 0 and self.merged:
                matmul_output = self.lora_B.weight @ self.lora_A.weight
                self.weight.data -= transpose(matmul_output.to(previous_dtype), self.fan_in_fan_out) * self.scaling
                self.merged = False

            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            # print('Jesus! output average')
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                result += self.lora_B(self.lora_A(self.lora_dropout(x.to(self.lora_A.weight.dtype)))) * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result

if is_bnb_available():
        
    class Linear4bit(bnb.nn.Linear4bit, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            merge_weights: bool = True,
            number_of_adapter_pre_layer: int = 1,
            layer: int = 1,
            proj: str = '',
            adapter_zoo = None,
            peft_config = None,
            **kwargs,
        ):
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=torch.bfloat16,
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights, number_of_adapter_pre_layer = number_of_adapter_pre_layer, layer = layer, proj = proj, peft_config = peft_config)
            
            
            self.lora_adapter_zoo = adapter_zoo
            self.activated_adapter_num = number_of_adapter_pre_layer
            self.selective_idx = None # selected adapter(s) for this layer
            self.ot_diversified_dispatcher = self.peft_config.ot_diversified_dispatcher
            self.transport_plan = torch.nn.Parameter(torch.zeros((1,1)), requires_grad=True)
            self.adaptive_ratio = self.peft_config.adaptive_ratio
            self.adaptive_ratio_decay = self.peft_config.adaptive_ratio_decay
            self.random_ratio = self.peft_config.random_ratio
            self.random_ratio_decay = self.peft_config.random_ratio_decay
            self.scaling_ratio = self.peft_config.scaling_ratio
            self.scaling_ratio_decay = self.peft_config.scaling_ratio_decay
            
            self.module_idx = peft_config.target_modules.index(proj)
            self.proj = proj

            # Actual trainable parameters
            if r > 0:
                # self.lora_A = nn.ModuleList()
                # self.lora_B = nn.ModuleList()
                # for _ in range(number_of_adapter_pre_layer):
                #     self.lora_A.append(nn.Linear(in_features, r, bias=False))
                #     self.lora_B.append(nn.Linear(r, out_features, bias=False))

                self.scaling = self.lora_alpha / (self.r)
                self.lora_scaling = torch.ones(number_of_adapter_pre_layer)
                self.lora_scaling = torch.nn.Parameter(self.lora_scaling.cuda().requires_grad_())
                nn.init.zeros_(self.lora_scaling)

                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
                
            # self.reset_parameters()
            
            # for i in range(self.number_of_adapter_pre_layer):
            #     if self.peft_config.insert_zero_lora and i == 0:
            #         nn.init.zeros_(self.lora_A[i].weight)
            #         nn.init.zeros_(self.lora_B[i].weight)
            #     else:
            #         nn.init.kaiming_uniform_(self.lora_A[i].weight, a=math.sqrt(5))
            #         nn.init.kaiming_uniform_(self.lora_B[i].weight, a=math.sqrt(5))

        # def reset_parameters(self):
        #     pass
        
        def load_selected_adapters_from_pool(self, input, sel_num = None, layer_idx = None, module_name = None, ratio = 0.2, random_ratio=1.0, inputs_embeds = None):
            self.selective_idx, A_param, B_param, routing_weights, used_experts, expert_mask = self.lora_adapter_zoo.select_best_for_input(input, self.number_of_adapter_pre_layer if sel_num == None else sel_num, layer_idx = layer_idx, module_name = module_name, ratio = ratio, random_ratio=random_ratio, inputs_embeds = inputs_embeds)
            # print('list(A_param.keys()): ', list(A_param.keys()))
            
            lora_A = {}
            lora_B = {}
            for i, idx in enumerate(used_experts.tolist()):
                new_lora_A = nn.Linear(A_param[idx].shape[1], A_param[idx].shape[0])
                new_lora_A.load_state_dict({"weight": A_param[idx]}, strict=False)
                # self.lora_A[i] = new_lora_A.cuda()
                lora_A[idx] = new_lora_A.cuda()
                # print('self.lora_A[idx].requires_grad', self.lora_A[idx].requires_grad)

                new_lora_B = nn.Linear(B_param[idx].shape[1], B_param[idx].shape[0])
                new_lora_B.load_state_dict({"weight": B_param[idx]}, strict=False)
                # self.lora_B[i] = new_lora_B.cuda()
                lora_B[idx] = new_lora_B.cuda()
                
            return routing_weights, used_experts, expert_mask, lora_A, lora_B

        def forward(self, x: torch.Tensor, layer_wise_forward: Optional[bool] = False,
            active_layer: Optional[int] = 0,
            active_adapter: Optional[int] = 0,
            weighted_matrix = None,
            inputs_embeds = None):
            
            self.module_dict = self.lora_adapter_zoo.module_dict
            self.gid = self.module_dict[self.proj]
            routing_weights, used_experts, expert_mask, lora_A, lora_B = self.load_selected_adapters_from_pool(x.detach(), self.peft_config.selective_num, self.layer, self.proj, self.adaptive_ratio, self.random_ratio, inputs_embeds)
            if self.training:
                self.adaptive_ratio *= self.adaptive_ratio_decay
                self.random_ratio *= self.random_ratio_decay
            self.activated_adapter_num = self.peft_config.selective_num
            return_selective_idx = self.selective_idx
            transport_plan = None
            
            for lora in lora_A:
                lora_A[lora].weight.requires_grad = False
                lora_A[lora].bias.requires_grad = False
            for lora in lora_B:
                lora_B[lora].weight.requires_grad = False
                lora_B[lora].bias.requires_grad = False
                
            result = super().forward(x)
            
            # print('routing_weights.shape: ', routing_weights.shape)
            # if self.peft_config.allow_empty_lora:
            #     # result *= routing_weights[:,:,0:1]
            #     routing_weights = routing_weights[:,1:]
                
            routing_weights *= self.scaling_ratio
            
            adapted_result = torch.zeros(
                result.shape, dtype=result.dtype, device=result.device
            )
            
            # print('result.shape: ', result.shape)

            if self.disable_adapters:
                return result
            elif self.r > 0:
                # if not torch.is_autocast_enabled():
                #     expected_dtype = result.dtype

                #     if x.dtype != torch.float32:
                #         x = x.float()
                #     for i in range(self.activated_adapter_num):
                #         if routing_weights[:,:,i:i+1].shape[-1] != 0:
                #             result += self.lora_B[i](self.lora_A[i](self.lora_dropout(x))).to(expected_dtype) * routing_weights[:,:,i:i+1]
                # else:
                #     for i in range(self.activated_adapter_num):
                #         if routing_weights[:,:,i:i+1].shape[-1] != 0:
                #             result += self.lora_B[i](self.lora_A[i](self.lora_dropout(x))) * routing_weights[:,:,i:i+1]
                
                # print('routing_weights: ',)
                # print(routing_weights)
                       
                # print('x.shape: ', x.shape)     
                for expert_idx in used_experts.tolist():
                    lora_a = lora_A[expert_idx]
                    lora_b = lora_B[expert_idx]
                    i = used_experts.tolist().index(expert_idx)
                    idx, top_x = torch.where(expert_mask[expert_idx])
                    # print(f'idx {idx} top_x {top_x}')
                    # current_state = x[None, top_x].reshape(-1, hidden_dim)
                    current_state = x[top_x]
                    # print('current_state.shape: ', current_state.shape)
                    # print('routing_weights[top_x, idx, None].shape: ', routing_weights[top_x, idx, None].shape)
                    # print('routing_weights[top_x, idx, None]: ', routing_weights[top_x, idx, None])
                    # print('lora_b(lora_a(self.lora_dropout(current_state))).shape: ', lora_b(lora_a(self.lora_dropout(current_state))).shape)
                    
                    current_hidden_states = lora_b(lora_a(self.lora_dropout(current_state))) * routing_weights[top_x, idx, None].unsqueeze(2)
                    
                    # print('current_hidden_states.shape: ', current_hidden_states.shape)
                    
                    adapted_result.index_add_(0, top_x, current_hidden_states.to(x.dtype))
                    
                # print('adapted_result.shape: ', adapted_result.shape)        
                
            del lora_A, lora_B
                
            if self.training:
                self.scaling_ratio *= self.scaling_ratio_decay
                
            return result, transport_plan, return_selective_idx.tolist()
        
        
    class Pure_Linear4bit(bnb.nn.Linear4bit, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=torch.bfloat16,
                # has_fp16_weights=kwargs.get("has_fp16_weights", True),
                # memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                # threshold=kwargs.get("threshold", 0.0),
                # index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights, number_of_adapter_pre_layer = number_of_adapter_pre_layer, layer = layer, proj = proj, peft_config = peft_config)
            
            # Actual trainable parameters
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "lora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype) * self.scaling
                    result += output
                else:
                    output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
                    result += output
            return result