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

from .config import PeftConfig
from .peft_model import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)
from .tuners import LoraConfig, PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, BottleneckConfig, M_LoraConfig, MA_LoraConfig, Prefix_MA_LoraConfig, M_PrefixConfig, Forzen_MA_LoraConfig
from .utils import PromptLearningConfig

import torch

MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {
    "SEQ_CLS": PeftModelForSequenceClassification,
    "SEQ_2_SEQ_LM": PeftModelForSeq2SeqLM,
    "CAUSAL_LM": PeftModelForCausalLM,
    "TOKEN_CLS": PeftModelForTokenClassification,
}

PEFT_TYPE_TO_CONFIG_MAPPING = {
    "PROMPT_TUNING": PromptTuningConfig,
    "PREFIX_TUNING": PrefixTuningConfig,
    "P_TUNING": PromptEncoderConfig,
    "LORA": LoraConfig,
    "M_LORA": M_LoraConfig,
    "MA_LORA": MA_LoraConfig,
    "PREFIX_MA_LORA": Prefix_MA_LoraConfig,
    "FORZEN_MA_LORA": Forzen_MA_LoraConfig,
    "M_PREFIX": M_PrefixConfig,
    "BOTTLENECK": BottleneckConfig,
}

TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "switch_transformers": ["q", "v"]
}

TRANSFORMERS_MODELS_TO_BOTTLENECK_TARGET_MODULES_MAPPING = {
    "bloom": ["dense_h_to_4h", "dense_4h_to_h"],
    "gptj": ["fc_in", "fc_out"],
    "gpt_neo": ["c_fc", "c_proj"],
    "llama": ["gate_proj", "up_proj", "down_proj"],
    "opt": ["fc1", "fc2"],
    "chatglm": ["dense_h_to_4h", "dense_4h_to_h"],
}

TRANSFORMERS_MODELS_TO_ADAPTERP_TARGET_MODULES_MAPPING = {
    "bloom": ["dense_4h_to_h"],
    "gptj": ["fc_out"],
    "gpt_neo": ["c_proj"],
    "llama": ["down_proj"],
    "opt": ["fc2"],
    "chatglm": ["dense_4h_to_h"],
}

TRANSFORMERS_MODELS_TO_PARALLEL_TARGET_MODULES_MAPPING = {
    "bloom": ["query_key_value"],
    "gptj": ["q_proj", "v_proj", "k_proj"],
    "gpt_neo": ["q_proj", "v_proj", "k_proj"],
    "llama": ["q_proj", "v_proj", "k_proj"],
    "opt": ["q_proj", "v_proj", "k_proj"],
    "chatglm": ["query_key_value"],
}

from .tuners import (
    LoraModel,
    M_LoraModel,
    MA_LoraModel
)

PEFT_TYPE_TO_TUNER_MAPPING = {
    "LORA": LoraModel,
    "M_LORA": M_LoraModel,
    "MA_LORA": MA_LoraModel,
}



def get_peft_config(config_dict):
    """
    Returns a Peft config object from a dictionary.

    Args:
        config_dict (`Dict[str, Any]`): Dictionary containing the configuration parameters.
    """

    return PEFT_TYPE_TO_CONFIG_MAPPING[config_dict["peft_type"]](**config_dict)


def _prepare_prompt_learning_config(peft_config, model_config):
    if peft_config.num_layers is None:
        if "num_hidden_layers" in model_config:
            num_layers = model_config["num_hidden_layers"]
        elif "num_layers" in model_config:
            num_layers = model_config["num_layers"]
        elif "n_layer" in model_config:
            num_layers = model_config["n_layer"]
        else:
            raise ValueError("Please specify `num_layers` in `peft_config`")
        peft_config.num_layers = num_layers

    if peft_config.token_dim is None:
        if "hidden_size" in model_config:
            token_dim = model_config["hidden_size"]
        elif "n_embd" in model_config:
            token_dim = model_config["n_embd"]
        elif "d_model" in model_config:
            token_dim = model_config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `peft_config`")
        peft_config.token_dim = token_dim

    if peft_config.num_attention_heads is None:
        if "num_attention_heads" in model_config:
            num_attention_heads = model_config["num_attention_heads"]
        elif "n_head" in model_config:
            num_attention_heads = model_config["n_head"]
        elif "num_heads" in model_config:
            num_attention_heads = model_config["num_heads"]
        elif "encoder_attention_heads" in model_config:
            num_attention_heads = model_config["encoder_attention_heads"]
        else:
            raise ValueError("Please specify `num_attention_heads` in `peft_config`")
        peft_config.num_attention_heads = num_attention_heads

    if getattr(peft_config, "encoder_hidden_size", None) is None:
        setattr(peft_config, "encoder_hidden_size", token_dim)

    return peft_config


def _prepare_lora_config(peft_config, model_config):
    if peft_config.target_modules is None:
        if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
    # if len(peft_config.target_modules) == 1:
    #     peft_config.fan_in_fan_out = True
    #     peft_config.enable_lora = [True, False, True]
    if peft_config.inference_mode:
        peft_config.merge_weights = True
    return peft_config


def _prepare_bottleneck_config(peft_config, model_config):
    if peft_config.target_modules is None:
        if peft_config.use_parallel_adapter:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_PARALLEL_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_PARALLEL_TARGET_MODULES_MAPPING[model_config["model_type"]]
        elif peft_config.use_adapterp:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ADAPTERP_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_ADAPTERP_TARGET_MODULES_MAPPING[model_config["model_type"]]
        else:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_BOTTLENECK_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_BOTTLENECK_TARGET_MODULES_MAPPING[model_config["model_type"]]

    return peft_config
    


def get_peft_model(model, peft_config, custom_config = {}):
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """

    model_config = model.config.to_dict()
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    print('peft_config.peft_type:', peft_config.peft_type)
    print('type(peft_config):', type(peft_config))
    print('isinstance(peft_config, PromptLearningConfig):', isinstance(peft_config, PromptLearningConfig))
    print('peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys(): ', peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys())
    
    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
        if peft_config.peft_type == "LORA":
            peft_config = _prepare_lora_config(peft_config, model_config)
            return PeftModel(model, peft_config)
        elif peft_config.peft_type == "BOTTLENECK":
            peft_config = _prepare_bottleneck_config(peft_config, model_config)
            return PeftModel(model, peft_config)
    if not isinstance(peft_config, PromptLearningConfig):
        if peft_config.peft_type == "BOTTLENECK":
            peft_config = _prepare_bottleneck_config(peft_config, model_config)
        elif peft_config.peft_type in ["LORA", "M_LORA", "MA_LORA", "FORZEN_MA_LORA"]:
            print('peft_config:', peft_config)
            print('model_config:', model_config)
            for key in custom_config:
                setattr(peft_config, key, custom_config[key])
                
            if "num_hidden_layers" in model_config:
                peft_config.num_layers = model_config["num_hidden_layers"]
            elif "num_layers" in model_config:
                peft_config.num_layers = model_config["num_layers"]
            elif "n_layers" in model_config:
                peft_config.num_layers = model_config["n_layers"]
                
            peft_config = _prepare_lora_config(peft_config, model_config)
            
        elif peft_config.peft_type == "PREFIX_MA_LORA":
            # print('custom_config: ', custom_config)
            for key in custom_config:
                # print('key: ', key, ' custom_config[key]: ', custom_config[key])
                # print('peft_config.key: ', peft_config.key)
                # peft_config.key = custom_config[key]
                setattr(peft_config, key, custom_config[key])
                # print('peft_config.key: ', peft_config.key)
            print('peft_config:', peft_config)
            print('model_config:', model_config)
            peft_config = _prepare_lora_config(peft_config, model_config)
            try:
                peft_config = _prepare_prompt_learning_config(peft_config, model_config)
            except:
                if peft_config.num_layers == None:
                    peft_config.num_layers = model_config['n_layers']
                if peft_config.num_attention_heads == None:
                    peft_config.num_attention_heads = model_config['n_heads']
                peft_config = _prepare_prompt_learning_config(peft_config, model_config)
        elif peft_config.peft_type == "M_PREFIX":
            print('M_PREFIX mapping')
            # print('custom_config: ', custom_config)
            for key in custom_config:
                # print('key: ', key, ' custom_config[key]: ', custom_config[key])
                # print('peft_config.key: ', peft_config.key)
                # peft_config.key = custom_config[key]
                setattr(peft_config, key, custom_config[key])
                # print('peft_config.key: ', peft_config.key)
            print('peft_config:', peft_config)
            print('model_config:', model_config)
            # peft_config = _prepare_lora_config(peft_config, model_config)
            peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    else:
        print('get_peft_model 3')
        for key in custom_config:
                setattr(peft_config, key, custom_config[key])
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
            
        print('peft_config: ', peft_config)
        
    
    print('MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type]: ', MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type])
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config)


def inject_adapter_in_model(
    peft_config: PeftConfig, model: torch.nn.Module, adapter_name: str = "default"
) -> torch.nn.Module:
    r"""
    A simple API to create and inject adapter in-place into a model. Currently the API does not support prompt learning
    methods and adaption prompt. Make sure to have the correct `target_names` set in the `peft_config` object. The API
    calls `get_peft_model` under the hood but would be restricted only to non-prompt learning methods.

    Args:
        peft_config (`PeftConfig`):
            Configuration object containing the parameters of the Peft model.
        model (`torch.nn.Module`):
            The input model where the adapter will be injected.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
    """
    # if peft_config.is_prompt_learning() or peft_config.is_adaption_prompt():
    #     raise ValueError("`create_and_replace` does not support prompt learning and adaption prompt yet.")

    if peft_config.peft_type not in PEFT_TYPE_TO_TUNER_MAPPING.keys():
        raise ValueError(
            f"`inject_adapter_in_model` does not support {peft_config.peft_type} yet. Please use `get_peft_model`."
        )

    tuner_cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]

    # By instantiating a peft model we are injecting randomly initialized LoRA layers into the model's modules.
    # peft_model = tuner_cls(model = model, config = peft_config, adapter_name=adapter_name)
    peft_model = tuner_cls(peft_config, model)

    return peft_model.model