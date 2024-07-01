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

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftType, transpose
from ..config import PeftConfig, PromptLearningConfig

from copy import deepcopy, copy

import ot

from ot_pytorch import sink, sink_stabilized

from dataclasses import dataclass, field


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class Prefix_MA_LoraConfig(PromptLearningConfig):
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
    selective_num: int = field(default=1, metadata={"help": "The number of A-B pairs"})
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
        default=False, metadata={"help": "whether to rely on instance-level information for token-level assignment"}
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
    
    # config for prefix tuning
    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"},
    )
    num_virtual_tokens: int = field(default=None, metadata={"help": "Number of virtual tokens"})
    num_prefix_set: int = field(default=8, metadata={"help": "Number of virtual tokens"})
    ot_diversified_prefix: bool = field(
        default=False,
        metadata={"help": "Whether to use ot to diversify assignment"},
    )
    curr_learning: bool = field(
        default=False,
        metadata={"help": "Whether to use ot to use additive modeling"},
    )
    hypernetwork: bool = field(
        default=False,
        metadata={"help": "Whether to use ot to use additive modeling"},
    )
    scale: int = field(default=24, metadata={"help": "scale for supernet"})
    additive_modeling: bool = field(
        default=False,
        metadata={"help": "Whether to use ot to use additive modeling"},
    )
    detached_training: bool = field(
        default=False,
        metadata={"help": "Whether to use ot to detached training"},
    )
    allow_empty_lora: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )
    insert_zero_lora: bool = field(
        default=False, metadata={"help": "whether do selective inference"}
    )

    def __post_init__(self):
        self.peft_type = PeftType.PREFIX_MA_LORA
