"""persona.modeling -- model/tokenizer loading, LoRA, seeds (PIPELINE_SPEC §5.1).

Qwen3.5 (transformers 5.x) is a hybrid model: some layers use full attention (q/k/v/o_proj), the others a gated
delta-rule linear attention (in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj); MLPs are gate/up/down_proj.
"All linear" LoRA therefore targets the token-mixing projections of both layer types plus the MLP.
Without the optional kernels (flash-linear-attention, causal_conv1d) transformers falls back to a reference
implementation, which is correct but slower; the smoke test reports seconds per step so this can be judged.
"""
from __future__ import annotations

import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import sdpa_attention

from . import config as C

sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False      # Windows: SDPA + GQA path blows up VRAM

LORA_ALL = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
            "gate_proj", "up_proj", "down_proj")
LORA_MIX_ONLY = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_name(key_or_name: str) -> str:
    return C.MODELS.get(key_or_name, key_or_name)


def load_tokenizer(name: str, padding_side: str = "right"):
    tok = AutoTokenizer.from_pretrained(model_name(name))
    tok.padding_side = padding_side
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str, device: str = "cuda", dtype=torch.bfloat16, grad_ckpt: bool = False):
    m = AutoModelForCausalLM.from_pretrained(model_name(name), dtype=dtype)
    m = m.to(device)
    if grad_ckpt:
        m.gradient_checkpointing_enable()
        m.config.use_cache = False
    return m


def add_lora(model, r: int = C.LORA["r"], alpha: int = C.LORA["alpha"], dropout: float = C.LORA["dropout"],
             targets=LORA_ALL):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets), bias="none",
                     task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    if getattr(model, "is_gradient_checkpointing", False) or getattr(model.config, "use_cache", True) is False:
        pm.enable_input_require_grads()
    return pm


def load_adapter(model, path: str, trainable: bool = False):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, path, is_trainable=trainable)


def merge_adapter(peft_model):
    return peft_model.merge_and_unload()


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def n_params(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, tr


def vram_gb() -> float:
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0
