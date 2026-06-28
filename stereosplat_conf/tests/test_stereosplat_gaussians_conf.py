"""Gaussian sanitization and confidence extraction."""
from __future__ import annotations

import pytest
import torch


def test_sanitize_gaussians_15d_cleans_nan(ss, device):
    g = torch.zeros(2, 10, 15, device=device)
    g[..., 14] = 0.7
    g[0, 0, 0] = float("nan")
    g[0, 1, 7:11] = 0.0  # bad quaternion norm
    out = ss.sanitize_gaussians_tensor(g)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert out.shape[-1] == 15
    assert out[0, 1, 7].item() == pytest.approx(1.0)


def test_sanitize_gaussians_14d_layout(ss, device):
    g = torch.ones(1, 4, 14, device=device)
    out = ss.sanitize_gaussians_tensor(g)
    assert out.shape[-1] == 14


def test_conf_from_render_pkg_squeeze_channel(ss, device):
    conf = torch.ones(1, 2, 1, 4, 4, device=device) * 0.5
    pkg = {"conf": conf}
    out = ss.conf_from_render_pkg(pkg)
    assert out.shape == (1, 2, 4, 4)


def test_conf_from_render_pkg_4d(ss, device):
    conf = torch.ones(1, 2, 4, 4, device=device)
    out = ss.conf_from_render_pkg({"conf": conf})
    assert out.shape == (1, 2, 4, 4)


def test_conf_from_render_pkg_none():
    from stereosplat.models_lab.StereoSplat import stereosplat as ss

    assert ss.conf_from_render_pkg(None) is None
    assert ss.conf_from_render_pkg({}) is None
