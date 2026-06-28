"""Utilities for Stage2 conf-only fine-tuning (split conf_head + weight surgery)."""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

CV_HEAD_PREFIX = "costvolume_gs.gaussains_estimation_head."
VOL_DECODER_PREFIX = "volume_gs.gs_decoder."


def _split_conv_head_keys(
    sd: Dict[str, torch.Tensor],
    old_prefix: str,
    shared_prefix: str,
    rgb_prefix: str,
    conf_prefix: str,
) -> None:
    """Split unified gaussian_head into shared_hidden + rgb/conf output convs.

    Unified: Conv(128,15) -> GELU -> Conv(15,15)
    Split:   Conv(128,15) -> GELU -> Conv(15,14) + Conv(15,1)
    The second conv rows are copied verbatim so forward is bit-exact vs unified.
    """
    w0_key = old_prefix + "0.weight"
    if w0_key not in sd:
        return

    w0 = sd[w0_key]
    b0 = sd[old_prefix + "0.bias"]
    w2 = sd[old_prefix + "2.weight"]
    b2 = sd[old_prefix + "2.bias"]

    sd[shared_prefix + "0.weight"] = w0.clone()
    sd[shared_prefix + "0.bias"] = b0.clone()
    sd[rgb_prefix + "weight"] = w2[:14].clone()
    sd[rgb_prefix + "bias"] = b2[:14].clone()
    sd[conf_prefix + "weight"] = w2[14:15].clone()
    sd[conf_prefix + "bias"] = b2[14:15].clone()

    for k in list(sd.keys()):
        if k.startswith(old_prefix):
            del sd[k]


def _split_linear_decoder_keys(
    sd: Dict[str, torch.Tensor],
    old_prefix: str,
    rgb_key: str,
    conf_key: str,
    gs_dim: int = 15,
) -> None:
    w_key = old_prefix + "gs_decoder.weight"
    if w_key not in sd:
        return

    w = sd[w_key]
    b = sd[old_prefix + "gs_decoder.bias"]
    if w.shape[0] % gs_dim != 0:
        raise ValueError(
            f"Unexpected gs_decoder out dim {w.shape[0]} (not divisible by gs_dim={gs_dim})"
        )
    gpv = w.shape[0] // gs_dim
    rgb_rows = (gs_dim - 1) * gpv

    sd[rgb_key + ".weight"] = w[:rgb_rows].clone()
    sd[rgb_key + ".bias"] = b[:rgb_rows].clone()
    sd[conf_key + ".weight"] = w[rgb_rows:].clone()
    sd[conf_key + ".bias"] = b[rgb_rows:].clone()

    del sd[w_key]
    del sd[old_prefix + "gs_decoder.bias"]


def migrate_unified_to_split_conf_head(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a unified 15-channel head checkpoint to split conf_head layout."""
    sd = copy.deepcopy(state_dict)
    _split_conv_head_keys(
        sd,
        old_prefix=CV_HEAD_PREFIX + "gaussian_head.",
        shared_prefix=CV_HEAD_PREFIX + "shared_hidden.",
        rgb_prefix=CV_HEAD_PREFIX + "rgb_geom_head.",
        conf_prefix=CV_HEAD_PREFIX + "conf_head.",
    )
    _split_linear_decoder_keys(
        sd,
        old_prefix=VOL_DECODER_PREFIX,
        rgb_key=VOL_DECODER_PREFIX + "gs_rgb_decoder",
        conf_key=VOL_DECODER_PREFIX + "conf_decoder",
        gs_dim=15,
    )
    return sd


def freeze_all_except_conf_modules(model: nn.Module) -> Tuple[int, int, List[str]]:
    """Freeze entire model; only CV conf_head + volume conf_decoder remain trainable."""
    trainable_names: List[str] = []
    n_trainable = 0
    n_frozen = 0
    for name, param in model.named_parameters():
        train = (
            "gaussains_estimation_head.conf_head" in name
            or "gs_decoder.conf_decoder" in name
        )
        param.requires_grad_(train)
        if train:
            n_trainable += param.numel()
            trainable_names.append(name)
        else:
            n_frozen += param.numel()
    return n_trainable, n_frozen, trainable_names
