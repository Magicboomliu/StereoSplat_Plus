"""Shared fixtures for stereosplat unit tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)


@pytest.fixture(scope="session")
def ss():
    """Imported stereosplat module (heavy import, once per session)."""
    from stereosplat.models_lab.StereoSplat import stereosplat

    return stereosplat


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def rgb_pair(device):
    """Two RGB tensors [1,2,3,4,4] and matching depth/conf."""
    b, v, c, h, w = 1, 2, 3, 4, 4
    rgb_a = torch.zeros(b, v, c, h, w, device=device)
    rgb_b = torch.ones(b, v, c, h, w, device=device)
    depth_a = torch.full((b, v, h, w), 10.0, device=device)
    depth_b = torch.full((b, v, h, w), 20.0, device=device)
    conf_a = torch.full((b, v, h, w), 0.3, device=device)
    conf_b = torch.full((b, v, h, w), 0.8, device=device)
    return rgb_a, rgb_b, depth_a, depth_b, conf_a, conf_b


@pytest.fixture
def rgb6(device):
    """Six-view tensors for per-view adaptive fusion (KITTI layout)."""
    b, v, c, h, w = 1, 6, 3, 4, 4
    rgb_a = torch.zeros(b, v, c, h, w, device=device)
    rgb_b = torch.ones(b, v, c, h, w, device=device)
    depth_a = torch.full((b, v, h, w), 1.0, device=device)
    depth_b = torch.full((b, v, h, w), 2.0, device=device)
    conf_a = torch.linspace(0.1, 0.6, v, device=device).view(b, v, 1, 1).expand(b, v, h, w)
    conf_b = torch.linspace(0.2, 0.7, v, device=device).view(b, v, 1, 1).expand(b, v, h, w)
    return rgb_a, rgb_b, depth_a, depth_b, conf_a, conf_b
