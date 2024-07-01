# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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


@dataclass
class MA_LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        number_of_adapter_pre_layer: The number of A-B pairs.
    """

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
    input_based_adapter_selection: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    simple_instance_matching: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    simple_hidden_matching: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    random_routing: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    random_routing_inference: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    layer_to_lora: list = field(default_factory=list)
    allow_empty_lora: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    insert_zero_lora: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )

    def __post_init__(self):
        self.peft_type = PeftType.MA_LORA


import torch

def mlm(tensor, prob):
    tensor = tensor.detach().clone()

    rand = torch.rand(tensor.shape)
    mask_arr = (rand < prob)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            selection = torch.flatten(mask_arr[i,j].nonzero()).tolist()
            tensor[i, j, selection] = 0

    return tensor

class Adapter_Zoo(torch.nn.Module):

    def __init__(self, num_layer, emb_size, config):
        super().__init__()
        
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
        # self.num_layer = 32
        self.config = config
        
        
        self.target_modules = config.target_modules
        self.num_adapter_each_layer = config.number_of_adapter_pre_layer
        self.input_based_adapter_selection = config.input_based_adapter_selection
        self.simple_instance_matching = config.simple_instance_matching
        self.simple_hidden_matching = config.simple_hidden_matching
        self.random_routing = config.random_routing
        self.random_routing_inference = config.random_routing_inference
        
        
        if hasattr(config, "allow_empty_lora") and config.allow_empty_lora:
            self.lora_backbone_emb = torch.nn.Parameter(torch.ones((self.num_layer, 1, 1, self.emb_size)), requires_grad=True)
            nn.init.xavier_uniform_(self.lora_backbone_emb)
            self.allow_empty_lora = True
        else:
            self.allow_empty_lora = False
        
        
        print('self.num_layer:', self.num_layer)

        # Create a parameter tensor
        self.lora_adapter_emb = torch.nn.Parameter(torch.ones((self.num_layer, len(self.target_modules), self.num_adapter_each_layer, self.emb_size)), requires_grad=True)
        nn.init.xavier_uniform_(self.lora_adapter_emb)

        self.weighted_matrix = [[None for _ in range(len(self.target_modules))] for _ in range(self.num_layer)] # not sure about it
        self.m = nn.Softmax(dim=-1)

        self.lora_A_params_dict : Dict[str, OrderedDict[str, torch.Tensor]] = {
            a_id:{
                b_id:{
                    c_id: None for c_id in range(self.num_adapter_each_layer)
                    } for b_id in range(len(self.target_modules ))
                } for a_id in range(self.num_layer)
        }
        self.lora_B_params_dict : Dict[str, OrderedDict[str, torch.Tensor]] = {
            a_id:{
                b_id:{
                    c_id: None for c_id in range(self.num_adapter_each_layer)
                    } for b_id in range(len(self.target_modules ))
                } for a_id in range(self.num_layer)
        }


    # def forward(self, input, layer_idx, module_name, adapter_idx = None, selective_idx = None, inputs_embeds = None):
    def forward(self, input, layer_idx, module_name, selective_idx = None, inputs_embeds = None, training = False):
        
        inputs_embeds = inputs_embeds.float()
        
        if self.allow_empty_lora:
            backbone_emb = self.lora_backbone_emb[layer_idx,0,:]
            # print('backbone_emb.shape: ', backbone_emb.shape)
        
        # print('selective_idx == None: ', selective_idx == None)

        # the similarity are token-level and instance-level
        if selective_idx == None:
            
            lora_emb = self.lora_adapter_emb[layer_idx, self.target_modules.index(module_name),:]
            # print('lora_emb.shape: ', lora_emb.shape)
            
            if self.allow_empty_lora:
                lora_emb = torch.cat([backbone_emb, lora_emb], dim=0)
            
            if self.input_based_adapter_selection:
                if not self.simple_instance_matching:
                    pass
                else:
                    if (training and not self.random_routing) or (not training and not self.random_routing_inference):
                        if self.simple_hidden_matching:
                            inner_prod = self.m(torch.matmul(input.float(), lora_emb.reshape(-1, self.emb_size).permute(1,0))).mean(1).unsqueeze(1)
                        else:
                            inner_prod = self.m(torch.matmul(inputs_embeds.float(), lora_emb.reshape(-1, self.emb_size).permute(1,0))).mean(1).unsqueeze(1)
                    else:
                        inner_prod = self.m(torch.tensor(np.random.dirichlet([0.5]*lora_emb.shape[0], (input.shape[0], 1))).float()).cuda()
            else:
                if (training and not self.random_routing) or (not training and not self.random_routing_inference):
                    inner_prod = self.m(torch.matmul(input.float(), lora_emb.reshape(-1, self.emb_size).permute(1,0)))
                else:
                    inner_prod = self.m(torch.matmul(input.float(), lora_emb.reshape(-1, self.emb_size).permute(1,0)))
                    inner_prod = self.m(torch.tensor(np.random.dirichlet([0.5]*lora_emb.shape[0], (input.shape[0], input.shape[1]))).float()).cuda()
        else:
            # Prepare adapter embeddings
            adapter_emb_list = [self.lora_adapter_emb[i, j, k].unsqueeze(0) for i, j, k in selective_idx]
            adapter_emb = torch.cat(adapter_emb_list, dim=0)
            # print('adapter_emb.shape: ', adapter_emb.shape)
            
            if self.allow_empty_lora:
                adapter_emb = torch.cat([backbone_emb, adapter_emb], dim=0)
            
            if self.input_based_adapter_selection:
                if not self.simple_instance_matching:
                    pass
                else:
                    if (training and not self.random_routing) or (not training and not self.random_routing_inference):
                            if self.simple_hidden_matching:
                                inner_prod = self.m(torch.matmul(input.float(), adapter_emb.reshape(-1, self.emb_size).permute(1, 0))).mean(1).unsqueeze(1)
                            else:
                                inner_prod = self.m(torch.matmul(inputs_embeds.float(), adapter_emb.reshape(-1, self.emb_size).permute(1,0))).mean(1).unsqueeze(1)
                    else:
                        # print('use randomized assignment?')
                        inner_prod = self.m(torch.tensor(np.random.dirichlet([0.5]*inputs_embeds.shape[0], (1, adapter_emb.reshape(-1, self.emb_size).shape[1]))).float()).cuda()
            else:
                if (training and not self.random_routing) or (not training and not self.random_routing_inference):
                    inner_prod = self.m(torch.matmul(input.float(), adapter_emb.reshape(-1, self.emb_size).permute(1, 0)))
                else:
                    # print('use randomized assignment?')
                    inner_prod = self.m(torch.tensor(np.random.dirichlet([0.5]*inputs_embeds.shape[0], (inputs_embeds.shape[1], adapter_emb.reshape(-1, self.emb_size).shape[1]))).float()).cuda()

        self.weighted_matrix[layer_idx][self.target_modules.index(module_name)] = inner_prod
        
        
        # print('inner_prod.shape:', inner_prod.shape)
        if self.allow_empty_lora:
            print('the weight of backbone:', inner_prod[:,:,0])
            return inner_prod[:,:,1:]
        else:
            # if not 'Phi' in self.config.base_model_name_or_path: # and layer_idx in [0, 15, 31]:
            #     print('trigger!')
            #     return self.m(mlm(inner_prod, 0.2))
            #     # return mlm(inner_prod, 1.0)
            return inner_prod

    def select_best_for_input(self, input, sel_num, layer_idx = None, module_idx = None, ratio = 0.2, inputs_embeds = None):
        
        inputs_embeds = inputs_embeds.float()
        
        idx_tensor_ori = torch.arange(self.lora_adapter_emb.shape[0] * self.lora_adapter_emb.shape[1] * self.lora_adapter_emb.shape[2])
        idx_tensor = idx_tensor_ori.reshape(self.lora_adapter_emb.shape[0], self.lora_adapter_emb.shape[1], self.lora_adapter_emb.shape[2])

        if ratio < 1.0:
            start_idx = max(0, int(layer_idx - ratio * idx_tensor.shape[0]))
            end_idx = min(idx_tensor.shape[0], int(layer_idx + ratio * idx_tensor.shape[0] + 1))
            selected_idxs = idx_tensor[start_idx:end_idx, module_idx:module_idx + 1, :]
            lora_adapter_emb = self.lora_adapter_emb[start_idx:end_idx, module_idx:module_idx + 1, :]
            selected_idxs = selected_idxs.reshape(-1)
        else:
            lora_adapter_emb = self.lora_adapter_emb
            selected_idxs = idx_tensor_ori

        # Select adapters for each layer
        reshaped_emb = lora_adapter_emb.reshape(-1, self.emb_size)

        if self.input_based_adapter_selection:
            if not self.simple_instance_matching:
                pass
            else:
                if self.simple_hidden_matching:
                    all_inner_prod = self.m(torch.matmul(input.float(), reshaped_emb.permute(1, 0))).mean(1).unsqueeze(1)
                else:
                    all_inner_prod = self.m(torch.matmul(inputs_embeds.float(), reshaped_emb.permute(1, 0))).mean(1).unsqueeze(1)
        else:
            all_inner_prod = self.m(torch.matmul(input.float(), reshaped_emb.permute(1, 0)))
        
        all_inner_prod = all_inner_prod.sum(0, keepdim=True)
        all_inner_prod = all_inner_prod.sum(1, keepdim=True)    
        all_inner_prod = all_inner_prod.squeeze()
        # print('all_inner_prod.shape:', all_inner_prod.shape)
        topk_idxs = torch.topk(all_inner_prod, sel_num).indices
        
        shifted_idxs = []
        for idx in topk_idxs:
            shifted_idxs.append(selected_idxs[idx])
            
        res = []
        # num_layer, len(target_modules), num_adapter_each_layer
        for i in range(self.num_layer):
            for j in range(len(self.target_modules)):
                for k in range(self.num_adapter_each_layer):
                    if idx_tensor[i][j][k] in shifted_idxs:
                        res.append((i,j,k))
        return res
        

class MA_LoraModel(torch.nn.Module):
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
        # self.adapter_zoo = Adapter_Zoo(num_layer=32, num_adapter_each_layer=self.peft_config.number_of_adapter_pre_layer, emb_size=4096, dynamic_adapter_pool=self.peft_config.dynamic_adapter_pool, selective_num=self.peft_config.selective_num, pool_selective_training=self.peft_config.pool_selective_training, pool_selective_inference=self.peft_config.pool_selective_inference, target_modules=self.peft_config.target_modules)
        if 'llama' in self.peft_config.base_model_name_or_path:
            emb_size = 4096
        elif 'phi' in self.peft_config.base_model_name_or_path:
            emb_size = 2560
        elif 'gemma' in self.peft_config.base_model_name_or_path:
            emb_size = 2048
        elif 'Phi' in self.peft_config.base_model_name_or_path:
            emb_size = 3072
        else:
            emb_size = 2048
        self.adapter_zoo = Adapter_Zoo(num_layer=32, emb_size=emb_size, config = self.peft_config)
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias, self.peft_config.insert_zero_lora)

        self.forward = self.model.forward
        
        # print('for ma_lora --- ')
        # for n, p in self.named_modules():
        #     if isinstance(p, Linear):
        #         print(f'module {n}')

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
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
            # "peft_config": self.peft_config,
        }

        key_list = [key for key, _ in self.model.named_modules()]
        
        # print('key_list:', key_list)
        for key in key_list:
            # print('hasattr(self.peft_config, "layer_to_lora"):', hasattr(self.peft_config, "layer_to_lora"))
            # print('self.peft_config.layer_to_lora:', self.peft_config.layer_to_lora)
            # print('len(self.peft_config.layer_to_lora) > 0:', len(self.peft_config.layer_to_lora) > 0)
            # print('type(self.peft_config.layer_to_lora):', type(self.peft_config.layer_to_lora))
            if hasattr(self.peft_config, "layer_to_lora") and len(self.peft_config.layer_to_lora) > 0:
                # print('!:', any(['.' + str(item) + '.' in key for item in self.peft_config.layer_to_lora]))
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
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
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
                        # new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, layer=int(key.split('.')[2]), adapter_zoo=self.adapter_zoo, proj=key.split('.')[-1], peft_config = self.peft_config, **kwargs)
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    # print('currently used')
                    # currently used
                    # print('kwargs:', kwargs)
                    # print(f'key before initializing the layer {key}')
                    new_module = Linear(target.in_features, target.out_features, bias=bias, layer=int(key.split('.')[2] if not 'OLM' in self.peft_config.base_model_name_or_path else key.split('.')[-2]), adapter_zoo=self.adapter_zoo, proj=key.split('.')[-1], peft_config = self.peft_config, **kwargs)
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
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none", insert_zero_lora: bool = False) -> None:
    # print('MA_LORA mark_only_lora_as_trainable:', mark_only_lora_as_trainable)
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
        elif insert_zero_lora and ('lora_A.0.' in n or 'lora_B.0.' in n):
            p.requires_grad = False
        # else:
        #     print('trainable param: ', n)
        
    # print('bias:', bias)
    
    # print('all trainable params:')
    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(n)
            
    # print('---end---')
    
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




class Linear(nn.Linear, LoraLayer):
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
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights, number_of_adapter_pre_layer = number_of_adapter_pre_layer, layer = layer, proj = proj, peft_config = peft_config)

        self.fan_in_fan_out = fan_in_fan_out
        self.lora_adapter_zoo = adapter_zoo
        self.activated_adapter_num = number_of_adapter_pre_layer
        self.selective_idx = None # selected adapter(s) for this layer
        self.ot_diversified_dispatcher = self.peft_config.ot_diversified_dispatcher
        self.transport_plan = torch.nn.Parameter(torch.zeros((1,1)), requires_grad=True)
        self.adaptive_ratio = self.peft_config.adaptive_ratio
        self.adaptive_ratio_decay = self.peft_config.adaptive_ratio_decay
        
        self.module_idx = peft_config.target_modules.index(proj)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.ModuleList()
            self.lora_B = nn.ModuleList()
            for _ in range(number_of_adapter_pre_layer):
                self.lora_A.append(nn.Linear(in_features, r, bias=False))
                self.lora_B.append(nn.Linear(r, out_features, bias=False))
            # self.scaling = self.lora_alpha / (self.r*number_of_adapter_pre_layer)

            self.scaling = self.lora_alpha / (self.r)
            self.lora_scaling = torch.ones(number_of_adapter_pre_layer)
            self.lora_scaling = torch.nn.Parameter(self.lora_scaling.cuda().requires_grad_())
            nn.init.zeros_(self.lora_scaling)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            
        self.reset_parameters()
        
        for i in range(self.number_of_adapter_pre_layer):
            if self.peft_config.insert_zero_lora and i == 0:
                nn.init.zeros_(self.lora_A[i].weight)
                nn.init.zeros_(self.lora_B[i].weight)
            else:
                nn.init.kaiming_uniform_(self.lora_A[i].weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lora_B[i].weight, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B[i].weight)
        # for i in range(self.number_of_adapter_pre_layer):
        #     # print(i,' adapter weight')
        #     print(self.lora_A[i].weight)

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        if self.peft_config.dynamic_adapter_pool:
            for i in range(self.number_of_adapter_pre_layer):
                self.lora_adapter_zoo.lora_A_params_dict[self.layer][self.module_idx][i] =  deepcopy(self.lora_A[i].state_dict())
                self.lora_adapter_zoo.lora_B_params_dict[self.layer][self.module_idx][i] =  deepcopy(self.lora_B[i].state_dict())

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        # print('hasattr(self, "lora_A"):', hasattr(self, "lora_A"))
        # if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
        # print('self.layer:', self.layer)
        # for i in range(self.number_of_adapter_pre_layer):
        #     nn.init.kaiming_uniform_(self.lora_A[i].weight, a=math.sqrt(5))
        #     nn.init.zeros_(self.lora_B[i].weight)
        # for i in range(self.number_of_adapter_pre_layer):
        #     print(i,' adapter weight')
        #     print(self.lora_A[i].weight)

    def load_selected_adapters_from_pool(self, input, sel_num = None, layer_idx = None, module_idx = None, ratio = 0.2, inputs_embeds = None):
        self.selective_idx = self.lora_adapter_zoo.select_best_for_input(input, self.number_of_adapter_pre_layer if sel_num == None else sel_num, layer_idx = layer_idx, module_idx = module_idx, ratio = ratio, inputs_embeds = inputs_embeds)
        for idx,(i,j,k) in enumerate(self.selective_idx):
            # print(f'i,j,k {i},{j},{k}')
            self.lora_A[idx].load_state_dict(self.lora_adapter_zoo.lora_A_params_dict[i][j][k], strict=False)
            self.lora_B[idx].load_state_dict(self.lora_adapter_zoo.lora_B_params_dict[i][j][k], strict=False)

    def save_adapters_to_pool(self):
        if self.selective_idx == None:
            return
        # for i in range(self.activated_adapter_num):
        for idx,(i,j,k) in enumerate(self.selective_idx):
            self.lora_adapter_zoo.lora_A_params_dict[i][j][k] =  deepcopy(self.lora_A[idx].state_dict())
            self.lora_adapter_zoo.lora_B_params_dict[i][j][k] =  deepcopy(self.lora_B[idx].state_dict())


    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        # print(f'mode {mode} self.merge_weights {self.merge_weights} not self.merged {not self.merged}')
        if not mode and self.merge_weights and not self.merged and not isinstance(self.lora_B, nn.ModuleList):
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.lora_scaling
                )
            self.merged = True
        elif self.merge_weights and self.merged and not isinstance(self.lora_B, nn.ModuleList):
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.lora_scaling
                )
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor,         
        layer_wise_forward: Optional[bool] = False,
        active_layer: Optional[int] = 0,
        active_adapter: Optional[int] = 0,
        weighted_matrix = None,
        inputs_embeds = None): # useless parameter
        
        if self.peft_config.dynamic_adapter_pool:
            self.save_adapters_to_pool()
            self.activated_adapter_num = self.number_of_adapter_pre_layer
            if (self.training and self.peft_config.pool_selective_training):
                # print('self.adaptive_ratio: ', self.adaptive_ratio)
                # print(f'inputs_embeds.shape {inputs_embeds.shape} x.shape {x.shape}')
                self.load_selected_adapters_from_pool(x.detach(), None, self.layer, self.module_idx, self.adaptive_ratio, inputs_embeds)
                self.adaptive_ratio *= self.adaptive_ratio_decay
            elif not self.training and self.peft_config.pool_selective_inference:
                self.load_selected_adapters_from_pool(x.detach(), self.peft_config.selective_num, self.layer, self.module_idx, self.adaptive_ratio, inputs_embeds)
                self.activated_adapter_num = self.peft_config.selective_num
            return_selective_idx = self.selective_idx
        else:
            self.activated_adapter_num = self.number_of_adapter_pre_layer
            self.selective_idx = None
            return_selective_idx = []
            for k in range(self.activated_adapter_num):
                return_selective_idx.append((self.layer,self.module_idx,k))

        # print('active_layer:', active_layer, ' active_adapter:', active_adapter)

        previous_dtype = self.weight.dtype
        transport_plan = None
        
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                matmul_output = self.lora_B.weight @ self.lora_A.weight
                print('changed self.weight')
                self.weight.data -= transpose(matmul_output.to(previous_dtype), self.fan_in_fan_out) * self.scaling
                self.merged = False

            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            result = F.linear(x.to(self.weight.dtype), transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                dispatch_weights = self.lora_adapter_zoo(x.detach(), self.layer, self.proj, selective_idx = self.selective_idx, inputs_embeds = inputs_embeds.detach(), training = self.training)
                if self.ot_diversified_dispatcher:
                    bs, ts, num_adapter = dispatch_weights.shape
                    expanded = torch.reshape(dispatch_weights, (bs*ts, -1))
                    transport_plan = sink(-expanded, reg=1, cuda=True)
                    transport_plan = torch.reshape(transport_plan, dispatch_weights.shape)
                    for i in range(self.activated_adapter_num):
                        result += self.lora_B[i](self.lora_A[i](self.lora_dropout(x.to(self.lora_A[i].weight.dtype)))) * transport_plan[:,:,i:i+1]
                elif not layer_wise_forward or (layer_wise_forward and active_layer != self.layer):
                    for i in range(self.activated_adapter_num):
                        torch.cuda.empty_cache()
                        result += self.lora_B[i](self.lora_A[i](self.lora_dropout(x.to(self.lora_A[i].weight.dtype)))) * dispatch_weights[:,:,i:i+1]
                        torch.cuda.empty_cache()
                else:
                    result += self.lora_B[active_adapter](self.lora_A[active_adapter](self.lora_dropout(x.to(self.lora_A[active_adapter].weight.dtype)))) * self.scaling
                del dispatch_weights
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result, transport_plan, return_selective_idx

class MergedLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        if out_features % len(enable_lora) != 0:
            raise ValueError("The length of enable_lora must divide out_features")
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
            self.lora_B = nn.Conv1d(
                r * sum(enable_lora),
                out_features // len(enable_lora) * sum(enable_lora),
                kernel_size=1,
                groups=2,
                bias=False,
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = (
                    F.conv1d(
                        self.lora_A.weight.data.unsqueeze(0),
                        self.lora_B.weight.data,
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                self.weight.data += transpose(self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out)
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = (
                    F.conv1d(
                        self.lora_A.weight.data.unsqueeze(0),
                        self.lora_B.weight.data,
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                self.weight.data -= transpose(self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out)
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.r > 0 and self.merged and any(self.enable_lora):
                delta_w = (
                    F.conv1d(
                        self.lora_A.weight.data.unsqueeze(0),
                        self.lora_B.weight.data,
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                delta_w = delta_w.to(self.weight.dtype)
                self.weight.data -= transpose(self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out)
                self.merged = False
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                after_A = self.lora_A(self.lora_dropout(x.to(self.lora_A.weight.dtype)))
                after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
        result = result.to(previous_dtype)

        return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
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
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
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

    class MergedLinear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            enable_lora: List[bool] = [False],
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
            if out_features % len(enable_lora) != 0:
                raise ValueError("The length of enable_lora must divide out_features")
            self.enable_lora = enable_lora
            # Actual trainable parameters
            if r > 0 and any(enable_lora):
                self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
                self.lora_B = nn.Conv1d(
                    r * sum(enable_lora),
                    out_features // len(enable_lora) * sum(enable_lora),
                    kernel_size=1,
                    groups=2,
                    bias=False,
                )
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
                # Compute the indices
                self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
                self.lora_ind[enable_lora, :] = True
                self.lora_ind = self.lora_ind.view(-1)
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "lora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

        def zero_pad(self, x):
            result = x.new_zeros((*x.shape[:-1], self.out_features))
            result = result.view(-1, self.out_features)
            result[:, self.lora_ind] = x.reshape(
                -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
            )
            return result.view((*x.shape[:-1], self.out_features))

        def forward(self, x: torch.Tensor):
            result = super().forward(x)
            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype
                    if x.dtype != torch.float32:
                        x = x.float()
                    after_A = self.lora_A(self.lora_dropout(x))
                    after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                    output = self.zero_pad(after_B).to(expected_dtype) * self.scaling
                    result += output
                else:
                    after_A = self.lora_A(self.lora_dropout(x))
                    after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                    output = self.zero_pad(after_B) * self.scaling
                    result += output
            return result
