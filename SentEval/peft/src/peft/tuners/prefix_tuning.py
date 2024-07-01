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


from dataclasses import dataclass, field

import torch

from ..utils import PeftType, PromptLearningConfig

import torch.nn.functional as F

from ot_pytorch import sink, sink_stabilized

import ot
import numpy as np


@dataclass
class PrefixTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.PrefixEncoder`].

    Args:
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        prefix_projection (`bool`): Whether to project the prefix embeddings.
    """

    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"},
    )

    def __post_init__(self):
        self.peft_type = PeftType.PREFIX_TUNING


# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor
class PrefixEncoder(torch.nn.Module):
    r"""
    The torch.nn model to encode the prefix

    Args:
        config ([`PrefixTuningConfig`]): The configuration of the prefix encoder.

    Example::

        >>> from peft import PrefixEncoder, PrefixTuningConfig >>> config = PrefixTuningConfig(
                peft_type="PREFIX_TUNING", task_type="SEQ_2_SEQ_LM", num_virtual_tokens=20, token_dim=768,
                num_transformer_submodules=1, num_attention_heads=12, num_layers=12, encoder_hidden_size=768
            )
        >>> prefix_encoder = PrefixEncoder(config)


    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) --
            The embedding layer of the prefix encoder.
        - **transform** (`torch.nn.Sequential`) -- The
        two-layer MLP to transform the prefix embeddings if `prefix_projection` is `True`.
        - **prefix_projection** (`bool`) -- Whether to project the prefix embeddings.

    Input shape: (batch_size, num_virtual_tokens)

    Output shape: (batch_size, num_virtual_tokens, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        if self.prefix_projection and not config.inference_mode:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = torch.nn.Sequential(
                torch.nn.Linear(token_dim, encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            )
        else:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)

    def forward(self, prefix: torch.Tensor):
        # print('self.embedding.shape:', self.embedding.shape)
        print('prefix.shape:', prefix.shape)
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        print('past_key_values.shape:', past_key_values.shape)
        return past_key_values
 
 
 
def D_reg(G):
    # print('np.max(G, axis=0).shape:', np.max(G, axis=1).shape)
    return - np.sum(np.max(G, axis=1))

def d_D_reg(G):
    idx = np.argmax(G, axis=1)
    tmp = np.zeros(G.shape)
    for col_idx, i in enumerate(idx):
        tmp[col_idx][i] = - 1.0
    return tmp   



from typing import List

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models


class SuperNet(nn.Module):
    """LeNet Hypernetwork"""

    def __init__(
        self,
        prefix_len,
        token_dim,
        scale,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        
        self.ray_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim//scale),
            nn.ReLU(inplace=True),
            # nn.Linear(token_dim, token_dim),
        )

        for i in range(self.prefix_len):
            setattr(
                self,
                f"token_{i}_weights",
                nn.Sequential(
                    nn.Linear(token_dim//scale, token_dim)
                )
            )


    def forward(self, inputs_embeds):
        inputs_embeds = self.ray_mlp(inputs_embeds)
        
        # inputs_embeds
        token_weights = []
        for i in range(self.prefix_len):
            token_weights.append(getattr(self, f"token_{i}_weights")(
                inputs_embeds
            ).unsqueeze(1)) 
        token_weights = torch.cat(token_weights, dim=1).mean(2).squeeze(2)
        print('token_weights.shape:', token_weights.shape)
        return token_weights
    
    
class Mixture_PrefixEncoder(torch.nn.Module):
    r"""
    The torch.nn model to encode the prefix

    Args:
        config ([`PrefixTuningConfig`]): The configuration of the prefix encoder.

    Example::

        >>> from peft import PrefixEncoder, PrefixTuningConfig >>> config = PrefixTuningConfig(
                peft_type="PREFIX_TUNING", task_type="SEQ_2_SEQ_LM", num_virtual_tokens=20, token_dim=768,
                num_transformer_submodules=1, num_attention_heads=12, num_layers=12, encoder_hidden_size=768
            )
        >>> prefix_encoder = PrefixEncoder(config)


    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) --
            The embedding layer of the prefix encoder.
        - **transform** (`torch.nn.Sequential`) -- The
        two-layer MLP to transform the prefix embeddings if `prefix_projection` is `True`.
        - **prefix_projection** (`bool`) -- Whether to project the prefix embeddings.

    Input shape: (batch_size, num_virtual_tokens)

    Output shape: (batch_size, num_virtual_tokens, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        self.num_prefix_set = config.num_prefix_set
        self.ot_diversified_prefix = config.ot_diversified_prefix
        self.alpha = 1.0
        self.alpha_decay = 0.995
        self.curr_learning = config.curr_learning
        self.hypernetwork = config.hypernetwork
        
        print('self.num_prefix_set:', self.num_prefix_set)
        print('config:', config)
        
        if not config.hypernetwork:
            if config.additive_modeling:
                if self.prefix_projection and not config.inference_mode:
                    self.base_embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
                else:
                    self.base_embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)
                self.lora_prefix_scaling = torch.ones(1)
                self.lora_prefix_scaling = torch.nn.Parameter(self.lora_prefix_scaling.cuda().requires_grad_())
                torch.nn.init.zeros_(self.lora_prefix_scaling)
                self.lora_prefix_scaling.requires_grad = True
                    
            
            self.embeddings = torch.nn.ModuleList()
            for _ in range(self.num_prefix_set):
                if self.prefix_projection and not config.inference_mode:
                    # Use a two-layer MLP to encode the prefix
                    embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
                else:
                    embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)
                self.embeddings.append(embedding)   
                
            self.lora_prefix_emb = torch.nn.Parameter(torch.ones((self.num_prefix_set, token_dim)), requires_grad=True)
        else:
            self.lora_supernet = SuperNet(num_virtual_tokens, token_dim if self.prefix_projection else num_layers * 2 * token_dim, config.scale)
            
        if self.prefix_projection:
            # self.prefix_transform = torch.nn.Sequential(
            #         # torch.nn.Linear(token_dim, encoder_hidden_size),
            #         # torch.nn.Tanh(),
            #         # torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            #         torch.nn.Linear(token_dim, num_layers * 2 * token_dim),
            #         # torch.nn.Tanh(),
            #         # torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            #     )
            
            self.prefix_transform = torch.nn.Sequential(
                    # torch.nn.Linear(token_dim, encoder_hidden_size),
                    # torch.nn.Tanh(),
                    # torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
                    torch.nn.Linear(token_dim, token_dim//24),
                    torch.nn.Tanh(),
                    torch.nn.Linear(token_dim//24, num_layers * 2 * token_dim),
                )

        # assert torch.sum(torch.abs(self.embeddings[0].weight.data - self.embeddings[1].weight.data)).numpy() == 0
        
    def select_prefix(self, embedding):
        # linear_lora_prefix_embedding = [self.linear(lora_prefix_embedding.weight.cuda()) for lora_prefix_embedding in self.embeddings]
        alpha_p = 1.0
        print('input embedding.shape: ', embedding.shape)
        # print(f'embedding.shape {embedding.shape} self.lora_prefix_emb[i:i+1].shape {self.lora_prefix_emb[0].shape}')
        similarity_matrix = [torch.bmm(F.normalize(input=embedding, p=2, dim=2),
                                    F.normalize(input=self.lora_prefix_emb[i:i+1].repeat(embedding.shape[0], embedding.shape[1], 1),
                                                p=2, dim=0).permute(0, 2, 1)) for i in range(self.num_prefix_set)]
        # print(f'similarity_matrix[0].shape {similarity_matrix[0].shape}')
        s_t = [alpha_p * F.normalize(input=sim.clamp(min=0), p=2, dim=2) for sim in similarity_matrix]
        alpha = [F.softmax(input=s_t[i], dim=1) for i in range(len(s_t))]
        a_v = [torch.bmm(alpha[i].permute(0, 2, 1), embedding) for i in range(len(alpha))]
        s_t_i = [F.cosine_similarity(self.lora_prefix_emb[i], a_v[i], dim=2).mean(1).unsqueeze(1) for i in range(self.num_prefix_set)]   
        s_t = torch.cat(s_t_i, dim = 1)
        self.s_t = torch.nn.functional.softmax(s_t, dim = -1)
        
        print(f'self.s_t {self.s_t} {self.s_t.shape}')
        
        if self.ot_diversified_prefix:
            self.s_t = sink(-self.s_t, reg=1, cuda=True)       
            print('diversified self.s_t:', self.s_t)
        
        
    def forward(self, prefix: torch.Tensor, gating_tuning = False, prefix_tuning = False, choose_prefix = None, additive_modeling = False, base_forzen = False, true_pref = None, inputs_embeds = None):
        
        # assert not self.hypernetwork or not additive_modeling, 'at most one of self.hypernetwork and additive_modeling can be true'
        if self.hypernetwork:
            print('inputs_embeds.shape:', inputs_embeds.shape)
            if self.prefix_projection:
                prefix_tokens = self.lora_supernet(inputs_embeds)
                past_key_value = self.prefix_transform(prefix_tokens)
            else:
                past_key_value = self.lora_supernet(inputs_embeds)
                
            print('output of supernet: past_key_value.shape:', past_key_value.shape)
            return past_key_value
        else:
            if additive_modeling:
                if self.prefix_projection:
                    base = self.prefix_transform(self.base_embedding(prefix))
                else:
                    base = self.base_embedding(prefix)
                    
            if additive_modeling and not base_forzen:
                
                print('only for testing why additive modeling works not well')
                past_key_value = self.embeddings[0](prefix)
                print('base.detach() + past_key_values shape:', (base.detach() + past_key_value).shape)
                return base
            
            if true_pref is not None:
                true_pref = true_pref.unsqueeze(2).unsqueeze(3)
                print('true_pref.shape:', true_pref.shape)
                
            if choose_prefix == None:
                past_key_values = []
                for pi in range(self.num_prefix_set):
                    if self.prefix_projection:
                        prefix_tokens = self.embeddings[pi](prefix)
                        past_key_value = self.prefix_transform(prefix_tokens)
                    else:
                        past_key_value = self.embeddings[pi](prefix)
                    past_key_values.append(past_key_value.unsqueeze(1))
                past_key_values = torch.cat(past_key_values, dim = 1)
                
                self.s_t = torch.nn.functional.softmax(self.s_t, dim = -1)
                
                assert not gating_tuning or not prefix_tuning, 'at most one of _tuning can be true' 
                print('self.s_t.unsqueeze(2).unsqueeze(3).shape:', self.s_t.unsqueeze(2).unsqueeze(3).shape)
                print('past_key_values.shape:', past_key_values.shape)
                if not gating_tuning and not prefix_tuning:
                    print('training gating and prefix jointly mode')
                    if self.curr_learning and true_pref is not None:
                        print('use updated s_t')
                        past_key_values = torch.mul(self.s_t.unsqueeze(2).unsqueeze(3) + self.alpha * true_pref, past_key_values).sum(1)
                        self.alpha *= self.alpha_decay
                    else:
                        past_key_values = torch.mul(self.s_t.unsqueeze(2).unsqueeze(3), past_key_values).sum(1)
                elif gating_tuning:
                    print('only training gating')
                    if self.curr_learning and true_pref is not None:
                        print('use updated s_t')
                        past_key_values = torch.mul(self.s_t.unsqueeze(2).unsqueeze(3) + self.alpha * true_pref, past_key_values.detach()).sum(1)
                        self.alpha *= self.alpha_decay
                    else:
                        past_key_values = torch.mul(self.s_t.unsqueeze(2).unsqueeze(3), past_key_values.detach()).sum(1)
                elif prefix_tuning:
                    print('only training all prefixes')
                    if self.curr_learning and true_pref is not None:
                        print('use updated s_t')
                        past_key_values = torch.mul(self.s_t.unsqueeze(2).unsqueeze(3).detach() + self.alpha * true_pref, past_key_values).sum(1)
                    else:
                        past_key_values = torch.mul(self.s_t.unsqueeze(2).unsqueeze(3).detach(), past_key_values).sum(1)

            else:
                print(f'independently train one specific prefix {choose_prefix}')
                if self.prefix_projection:
                    prefix_tokens = self.embeddings[choose_prefix](prefix)
                    past_key_values = self.prefix_transform(prefix_tokens)
                else:
                    past_key_values = self.embeddings[choose_prefix](prefix)
            
            if not additive_modeling:
                print('final past_key_values.shape:', past_key_values.shape)
                return past_key_values
            elif additive_modeling and base_forzen:
                print('self.lora_prefix_scaling: ', self.lora_prefix_scaling)
                return base.detach() + past_key_values * self.lora_prefix_scaling