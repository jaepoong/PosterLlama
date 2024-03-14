import os
import logging
import contextlib

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig,AutoTokenizer,AutoModel,LlamaForCausalLM,LlamaTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import Dinov2Model
from src.model.eva_vit import create_eva_vit_g


from src.common.utils import download_cached_file,is_url


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision, freeze
    ):
        logging.info('Loading VIT')

        assert model_name == "eva_clip_g" or model_name == "dino_v2", "vit model must be eva_clip_g for current version of MiniGPT-4"
        if model_name =="eva_clip_g":
            if not freeze: # True, False
                precision = "fp32"  # fp16 is not for training

            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )

            ln_vision = LayerNorm(visual_encoder.num_features)

            if freeze:
                for name, param in visual_encoder.named_parameters():
                    param.requires_grad = False
                visual_encoder = visual_encoder.eval()
                visual_encoder.train = disabled_train
                for name, param in ln_vision.named_parameters():
                    param.requires_grad = False
                ln_vision = ln_vision.eval()
                ln_vision.train = disabled_train
                logging.info("freeze vision encoder")

            logging.info('Loading VIT Done')
            return visual_encoder, ln_vision
        elif model_name =="dino_v2":
            visual_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base")
            if freeze:
                for name, param in visual_encoder.named_parameters():
                    param.requires_grad = False
                visual_encoder = visual_encoder.eval()
                visual_encoder.train = disabled_train
                logging.info("freeze vision encoder")

            logging.info('Loading Dino-v2 Done')
            return visual_encoder, ""
        
            

    def init_llm(self, llama_model_path, low_resource=False, low_res_device=0, lora_r=0,
                 lora_target_modules=["q_proj","v_proj"], **lora_kargs):
        logging.info('Loading LLAMA')
        llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
        llama_tokenizer.pad_token = "$$"

        if low_resource:
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                use_cache=False,
                #device_map="cpu"
            )
        else:
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                use_cache=False
            )

        if lora_r > 0:
            llama_model = prepare_model_for_int8_training(llama_model)
            loraconfig = LoraConfig(
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
                **lora_kargs
            )
            llama_model = get_peft_model(llama_model, loraconfig)

            llama_model.print_trainable_parameters()

        else:
            for name, param in llama_model.named_parameters():
                param.requires_grad = False
        logging.info('Loading LLAMA Done')
        return llama_model, llama_tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)
        return msg

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)