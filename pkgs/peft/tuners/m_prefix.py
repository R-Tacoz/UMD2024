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


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class M_PrefixConfig(PromptLearningConfig):
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
    detached_training: bool = field(
        default=False,
        metadata={"help": "Whether to use ot to detached training"},
    )
    additive_modeling: bool = field(
        default=False,
        metadata={"help": "Whether to use ot to use additive modeling"},
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

    def __post_init__(self):
        self.peft_type = PeftType.M_PREFIX
