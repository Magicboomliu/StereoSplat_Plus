"""Pixel / soft confidence fusion helpers."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_fuse_renders_by_conf_pixelwise_prefers_high_conf_b(ss, rgb_pair):
    rgb_a, rgb_b, depth_a, depth_b, conf_a, conf_b = rgb_pair
    frgb, fdepth, fconf = ss.fuse_renders_by_conf_pixelwise(
        rgb_a, rgb_b, depth_a, depth_b, conf_a, conf_b,
    )
    assert torch.allclose(frgb, rgb_b)
    assert torch.allclose(fdepth, depth_b)
    assert torch.allclose(fconf, conf_b)


def test_fuse_renders_by_conf_margin_tie_prefers_a(ss, rgb_pair):
    rgb_a, rgb_b, depth_a, depth_b, conf_a, conf_b = rgb_pair
    conf_a = torch.full_like(conf_a, 0.5)
    conf_b = torch.full_like(conf_b, 0.5)
    frgb, _, _ = ss.fuse_renders_by_conf_pixelwise(
        rgb_a, rgb_b, depth_a, depth_b, conf_a, conf_b, conf_fusion_margin=0.0,
    )
    assert torch.allclose(frgb, rgb_a)


def test_fuse_renders_soft_conf_weighted_tie_near_mv(ss, rgb_pair):
    rgb_a, rgb_b, _, _, conf_a, conf_b = rgb_pair
    conf_a = torch.full_like(conf_a, 0.4)
    conf_b = torch.full_like(conf_b, 0.4)
    fused, w_mv = ss.fuse_renders_soft_conf_weighted(
        rgb_a, rgb_b, conf_a, conf_b, temperature=50.0, tie_logit_mv=4.595,
    )
    assert fused.shape == rgb_a.shape
    assert w_mv is not None
    assert w_mv.mean().item() > 0.9


def test_fuse_renders_per_view_adaptive_uses_margins(ss, rgb6):
    rgb_a, rgb_b, depth_a, depth_b, conf_a, conf_b = rgb6
    frgb, fdepth, fconf = ss.fuse_renders_per_view_adaptive(
        rgb_a, rgb_b, depth_a, depth_b, conf_a, conf_b,
        first_margin=999.0,
        center_margin=0.0,
        last_margin=0.0,
    )
    assert frgb.shape == rgb_a.shape
    assert fdepth.shape == depth_a.shape
    assert fconf.shape == conf_a.shape
    # first stereo pair (views 4,5) should stay on base due to huge first_margin
    assert torch.allclose(frgb[:, 4:6], rgb_a[:, 4:6])


def test_pixel_fuse_renders_legacy_mode(ss, rgb_pair):
    rgb_a, rgb_b, depth_a, depth_b, conf_a, conf_b = rgb_pair
    frgb, fdepth, fconf = ss.pixel_fuse_renders(
        rgb_a, rgb_b, depth_a, depth_b, conf_a, conf_b, fusion_mode="legacy",
    )
    assert torch.allclose(frgb, rgb_b)


def test_fuse_renders_eval_mode_hard(ss, rgb_pair):
    rgb_a, rgb_b, depth_a, depth_b, conf_a, conf_b = rgb_pair
    sup = SimpleNamespace(val_fusion_mode="hard")
    frgb, fdepth, fconf = ss.fuse_renders_eval_mode(
        rgb_a, rgb_b, depth_a, depth_b, conf_a, conf_b, fusion_sup_dict=sup,
    )
    assert torch.allclose(frgb, rgb_b)


def test_stereosplat_conf_eval_stats_buckets(ss, device):
    # 6 views interleaved stereo layout
    conf = torch.arange(24, device=device, dtype=torch.float32).view(1, 6, 2, 2)
    stats = ss.stereosplat_conf_eval_stats(conf)
    assert stats is not None
    assert "first_view_mean_conf_average" in stats
    assert "center_view_mean_conf_average" in stats
    assert "last_view_mean_conf_average" in stats
    assert stats["all_view_mean_conf_average"] == pytest.approx(conf.mean().item())


def test_key_view_psnr_margin_hinge_positive_when_worse(ss, device):
    err_ref = torch.full((1, 6), 0.01, device=device)
    err_tgt = torch.full((1, 6), 0.5, device=device)
    loss = ss._key_view_psnr_margin_hinge(
        err_ref, err_tgt, view_name="center", margin_db=0.0, slack_db=0.0,
    )
    assert loss.item() > 0.0
