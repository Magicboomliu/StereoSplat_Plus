"""Metrics, I/O helpers, and GT-error fusion."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image


def test_depth_metrics_absrel_scalar(ss, device):
    pred = torch.tensor([[[[2.0, 4.0], [4.0, 8.0]]]], device=device)
    gt = torch.tensor([[[[2.0, 4.0], [4.0, 8.0]]]], device=device)
    m = ss.depth_metrics_absrel_sqrel_rmse_log(pred, gt)
    assert m["AbsRel"].item() == pytest.approx(0.0, abs=1e-6)
    assert m["SqRel"].item() == pytest.approx(0.0, abs=1e-6)
    assert m["RMSE_log"].item() == pytest.approx(0.0, abs=1e-6)


def test_depth_metrics_absrel_per_view(ss, device):
    pred = torch.tensor([[[[3.0]], [[6.0]]]], device=device)
    gt = torch.tensor([[[[2.0]], [[4.0]]]], device=device)
    m = ss.depth_metrics_absrel_sqrel_rmse_log(pred, gt, per_view=True)
    assert m["AbsRel"].shape == (1, 2)
    assert m["AbsRel"][0, 0].item() == pytest.approx(0.5)
    assert m["AbsRel"][0, 1].item() == pytest.approx(0.5)


def test_compute_depth_stereo_mae_mse_even_views(ss, device):
    pred = torch.ones(1, 4, 2, 2, device=device)
    gt = torch.ones(1, 4, 2, 2, device=device) * 2.0
    l_mae, l_mse, r_mae, r_mse = ss.compute_depth_stereo_mae_mse(pred, gt)
    assert l_mae.item() == pytest.approx(1.0)
    assert r_mae.item() == pytest.approx(1.0)


def test_fuse_rgb_by_gt_error_picks_closer(ss, device):
    gt = torch.zeros(1, 1, 3, 2, 2, device=device)
    rgb1 = torch.zeros_like(gt)
    rgb2 = torch.ones_like(gt)
    m1, m2, fused = ss.fuse_rgb_by_gt_error(rgb1, rgb2, gt, metric="l1")
    assert torch.allclose(fused, rgb1)
    assert m1.sum() > m2.sum()


def test_fuse_rgb_by_gt_error_l2_metric(ss, device):
    gt = torch.zeros(1, 1, 3, 2, 2, device=device)
    rgb1 = torch.full_like(gt, 0.1)
    rgb2 = torch.full_like(gt, 0.9)
    _, _, fused = ss.fuse_rgb_by_gt_error(rgb1, rgb2, gt, metric="l2")
    assert fused.shape == gt.shape


def test_saved_into_json(tmp_path, ss):
    path = tmp_path / "m.json"
    ss.saved_into_json({"psnr": 30.5}, str(path))
    assert json.loads(path.read_text())["psnr"] == 30.5


def test_convert_a_numpy_to_uint8(ss):
    arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    out = ss.convert_a_numpy_to_uint8(arr)
    assert out.dtype == np.uint8
    assert out.tolist() == [0, 127, 255]


def test_convert_pil_to_tensor(ss):
    img = Image.new("RGB", (4, 2), color=(255, 128, 0))
    t = ss.convert_pil_to_tensor(img)
    assert t.shape == (1, 1, 3, 2, 4)
    assert t.max() <= 1.0
    assert t.min() >= 0.0


def test_metrics_mean_psnr_perfect(ss, device, monkeypatch):
    pred = torch.ones(1, 2, 3, 8, 8, device=device)
    gt = torch.ones_like(pred)
    mock_lpips = MagicMock()
    mock_lpips.return_value = torch.zeros(2, 1, 1, 1, device=device)
    monkeypatch.setattr(ss, "_cached_lpips", lambda _net, _dev: mock_lpips)
    out = ss.metrics_mean(pred, gt, lpips_net="alex")
    assert out["psnr"].item() > 80.0
    assert out["ssim"].item() > 0.99
    assert out["lpips"].item() == pytest.approx(0.0)
