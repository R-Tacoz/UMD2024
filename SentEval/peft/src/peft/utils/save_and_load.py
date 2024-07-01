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

from .config import PeftType
from typing import Optional

import torch
import os

WEIGHTS_NAME = "adapter_model.bin"
SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"


def get_peft_model_state_dict(model, state_dict=None):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    print('enter get_peft_model_state_dict')
    if state_dict is None:
        state_dict = model.state_dict()
    if model.peft_config.peft_type in [PeftType.LORA, PeftType.M_LORA, PeftType.MA_LORA, PeftType.PREFIX_MA_LORA, PeftType.M_PREFIX]:
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to directly with the state dict which is necessary when using DeepSpeed or FSDP
        try:
            bias = model.peft_config.bias
        except:
            bias = "none"
        if model.peft_config.peft_type == PeftType.PREFIX_MA_LORA:
            bias = "lora_prefix_only"
        # print('bias')
        if bias == "none" or model.peft_config.peft_type == PeftType.M_PREFIX:
            to_return = {k: state_dict[k] for k in state_dict if ("lora_" in k) or ("prefix_" in k)}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        elif bias == "lora_prefix_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
                if "prefix_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("prefix_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        
                
        else:
            raise NotImplementedError
        
        print(f'to_return after condition1 {list(to_return.keys())}')
    elif model.peft_config.peft_type == PeftType.BOTTLENECK:
        # return the state dict of the model with Bottleneck adapters
        bias = model.peft_config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "adapter_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "adapter_" in k or "bias" in k}
        elif bias == "adapter_only":
            to_return = {}
            for k in state_dict:
                if "adapter_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("adapter_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
    elif model.peft_config.peft_type == PeftType.FORZEN_MA_LORA:
        if model.peft_config.train_lm_head:
            to_return = {k: state_dict[k] for k in state_dict if ".adapter_zoo." in k or 'lm_head' in k}
        else:
            to_return = {k: state_dict[k] for k in state_dict if ".adapter_zoo." in k}
    else:
        to_return = {}
        if model.peft_config.inference_mode:
            prompt_embeddings = model.prompt_encoder.embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save()
        to_return["prompt_embeddings"] = prompt_embeddings
        
    if model.peft_config.peft_type == PeftType.PREFIX_MA_LORA or model.peft_config.peft_type == PeftType.M_PREFIX:
        if model.peft_config.hypernetwork:
            pass
        else:
            prompt_embeddings = model.prompt_encoder.embeddings
            
            if model.peft_config.additive_modeling:
                base_embedding = model.prompt_encoder.base_embedding
                to_return[f"base_embedding"] = base_embedding.weight
                # to_return['prefix_scaling'] = model.prompt_encoder.prefix_scaling.weight
            
            for i in range(len(prompt_embeddings)): # save each embedding seperately
                to_return[f"prompt_embedding_{i}"] = prompt_embeddings[i].weight
            
        # if model.peft_config.peft_type == PeftType.M_PREFIX:
        #     to_return[f"lora_prefix_emb"] = model.prompt_encoder.lora_prefix_emb.weight
            
        if model.peft_config.peft_type == PeftType.PREFIX_MA_LORA:
            to_return['lora_A_params_dict'] = model.base_model.adapter_zoo.lora_A_params_dict
            to_return['lora_B_params_dict'] = model.base_model.adapter_zoo.lora_B_params_dict
        
        
    # print('model.prompt_encoder.lora_prefix_emb: ', model.prompt_encoder.lora_prefix_emb)
    print('to_return')
    print(list(to_return.keys()))
    # print(to_return)
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                to_return[key] = value
    return to_return


def set_peft_model_state_dict(model, peft_model_state_dict):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    print('before model.load_state_dict')
    print('peft_model_state_dict: ', peft_model_state_dict)
    # print('model.load_state_dict:', model.load_state_dict)
    model.load_state_dict(peft_model_state_dict, strict=False)
    if model.peft_config.peft_type not in [PeftType.LORA, PeftType.M_LORA, PeftType.MA_LORA, PeftType.PREFIX_MA_LORA, PeftType.M_PREFIX, PeftType.BOTTLENECK, PeftType.FORZEN_MA_LORA]:
        if not (model.peft_config.peft_type == PeftType.M_PREFIX and model.peft_config.hypernetwork):
            model.prompt_encoder.embedding.load_state_dict(
                {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
            )
    elif model.peft_config.peft_type == PeftType.PREFIX_MA_LORA or model.peft_config.peft_type == PeftType.M_PREFIX:
        if not model.peft_config.hypernetwork:
            print('peft_model_state_dict: ', peft_model_state_dict)
            print('len(model.prompt_encoder.embeddings): ', len(model.prompt_encoder.embeddings))
            print('model.prompt_encoder: ', model.prompt_encoder)
            for i in range(len(model.prompt_encoder.embeddings)):
                model.prompt_encoder.embeddings[i].load_state_dict(
                    {"weight": peft_model_state_dict[f"prompt_embedding_{i}"]}, strict=True
                )
            if model.peft_config.additive_modeling:
                print('loading base_embedding')
                model.prompt_encoder.base_embedding.load_state_dict(
                    {"weight": peft_model_state_dict[f"base_embedding"]}, strict=True
                )
                # model.prompt_encoder.prefix_scaling.load_state_dict(
                #     {"weight": peft_model_state_dict[f"prefix_scaling"]}, strict=True
                # )
            
        if model.peft_config.peft_type == PeftType.PREFIX_MA_LORA:
            model.base_model.adapter_zoo.lora_A_params_dict = peft_model_state_dict["lora_A_params_dict"]
            model.base_model.adapter_zoo.lora_B_params_dict = peft_model_state_dict["lora_B_params_dict"]
            
            # try:
            # print('model.base_model.model.model.transformer.blocks: ')
            # print(model.base_model.model.model.transformer.blocks)
            # except:
            #     pass
            
            try:
                for layer in model.base_model.model.model.layers:
                    try:
                        layer.self_attn.q_proj.lora_adapter_zoo.lora_A_params_dict = peft_model_state_dict["lora_A_params_dict"]
                        layer.self_attn.q_proj.lora_adapter_zoo.lora_B_params_dict = peft_model_state_dict["lora_B_params_dict"]
                    except:
                        pass
                    
                    try:
                        layer.self_attn.v_proj.lora_adapter_zoo.lora_A_params_dict = peft_model_state_dict["lora_A_params_dict"]
                        layer.self_attn.v_proj.lora_adapter_zoo.lora_B_params_dict = peft_model_state_dict["lora_B_params_dict"]
                    except:
                        pass
                    
                    try:
                        layer.self_attn.k_proj.lora_adapter_zoo.lora_A_params_dict = peft_model_state_dict["lora_A_params_dict"]
                        layer.self_attn.k_proj.lora_adapter_zoo.lora_B_params_dict = peft_model_state_dict["lora_B_params_dict"]
                    except:
                        pass
                    
                    try:
                        layer.self_attn.o_proj.lora_adapter_zoo.lora_A_params_dict = peft_model_state_dict["lora_A_params_dict"]
                        layer.self_attn.o_proj.lora_adapter_zoo.lora_B_params_dict = peft_model_state_dict["lora_B_params_dict"]
                    except:
                        pass
                    
                    try:
                        # qkv_proj
                        layer.self_attn.qkv_proj.lora_adapter_zoo.lora_A_params_dict = peft_model_state_dict["lora_A_params_dict"]
                        layer.self_attn.qkv_proj.lora_adapter_zoo.lora_B_params_dict = peft_model_state_dict["lora_B_params_dict"]
                    except:
                        pass
                    
            except:
                for layer in model.base_model.model.model.transformer.blocks:
                    try:
                        layer.attn_out.lora_adapter_zoo.lora_A_params_dict = peft_model_state_dict["lora_A_params_dict"]
                        layer.attn_out.lora_adapter_zoo.lora_B_params_dict = peft_model_state_dict["lora_B_params_dict"]
                    except:
                        pass
    return model


def infer_device():
    if torch.cuda.is_available():
        torch_device = "cuda"
    elif is_xpu_available():
        torch_device = "xpu"
    elif is_npu_available():
        torch_device = "npu"
    else:
        torch_device = "cpu"
    return torch_device


def load_peft_weights(model_id: str, device: Optional[str] = None, **hf_hub_download_kwargs) -> dict:
    r"""
    A helper method to load the PEFT weights from the HuggingFace Hub or locally

    Args:
        model_id (`str`):
            The local path to the adapter weights or the name of the adapter to load from the HuggingFace Hub.
        device (`str`):
            The device to load the weights onto.
        hf_hub_download_kwargs (`dict`):
            Additional arguments to pass to the `hf_hub_download` method when loading from the HuggingFace Hub.
    """
    path = (
        os.path.join(model_id, hf_hub_download_kwargs["subfolder"])
        if hf_hub_download_kwargs.get("subfolder", None) is not None
        else model_id
    )

    if device is None:
        device = infer_device()

    if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
        filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
        use_safetensors = True
    elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
        filename = os.path.join(path, WEIGHTS_NAME)
        use_safetensors = False
    else:
        has_remote_safetensors_file = hub_file_exists(
            model_id,
            SAFETENSORS_WEIGHTS_NAME,
            revision=hf_hub_download_kwargs.get("revision", None),
            repo_type=hf_hub_download_kwargs.get("repo_type", None),
        )
        use_safetensors = has_remote_safetensors_file

        if has_remote_safetensors_file:
            # Priority 1: load safetensors weights
            filename = hf_hub_download(
                model_id,
                SAFETENSORS_WEIGHTS_NAME,
                **hf_hub_download_kwargs,
            )
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME, **hf_hub_download_kwargs)
            except EntryNotFoundError:
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} or {SAFETENSORS_WEIGHTS_NAME} is present at {model_id}."
                )

    if use_safetensors:
        adapters_weights = safe_load_file(filename, device=device)
    else:
        adapters_weights = torch.load(filename, map_location=torch.device(device))

    return adapters_weights

