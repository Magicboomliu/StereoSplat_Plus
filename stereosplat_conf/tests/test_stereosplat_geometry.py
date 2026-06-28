"""Geometry / pose / depth helpers in stereosplat.py."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch


def test_write_pose_to_json_roundtrip(tmp_path, ss):
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    out = tmp_path / "pose.json"
    ss.write_pose_to_json(pose, out)
    data = json.loads(out.read_text())
    assert np.allclose(data["pose"], pose)


def test_write_pose_to_json_rejects_bad_shape(ss):
    with pytest.raises(ValueError, match="Expected pose shape"):
        ss.write_pose_to_json(np.eye(3), "/tmp/x.json")


def test_build_w2i_from_c2w_shape(ss, device):
    b, v = 2, 3
    c2w = torch.eye(4, device=device).view(1, 1, 4, 4).expand(b, v, 4, 4).clone()
    c2w[:, :, :3, 3] = torch.tensor([0.0, 0.0, 5.0], device=device)
    k = torch.eye(3, device=device).view(1, 1, 3, 3).expand(b, v, 3, 3).clone()
    k[:, :, 0, 0] = 500.0
    k[:, :, 1, 1] = 500.0
    k[:, :, 0, 2] = 32.0
    k[:, :, 1, 2] = 16.0
    w2i = ss.build_w2i_from_c2w(c2w, k)
    assert w2i.shape == (b, v, 4, 4)


def test_make_poses_relative_to_reference_is_identity(ss, device):
    poses = torch.eye(4, device=device).view(1, 1, 4, 4).expand(1, 4, 4, 4).clone()
    ref = poses[0, 2]
    rel = ss.make_poses_relative_to_reference_c2w(poses, ref)
    for v in range(4):
        assert torch.allclose(rel[0, v], torch.eye(4, device=device), atol=1e-5)


def test_make_poses_relative_batch_ref(ss, device):
    b, v = 2, 2
    poses = torch.eye(4, device=device).view(1, 1, 4, 4).expand(b, v, 4, 4).clone()
    ref = torch.eye(4, device=device).unsqueeze(0).expand(b, 4, 4).clone()
    rel = ss.make_poses_relative_to_reference_c2w(poses, ref)
    assert rel.shape == (b, v, 4, 4)
    assert torch.allclose(rel, torch.eye(4, device=device).view(1, 1, 4, 4))


def test_compute_depth_mae_mse_masks_invalid(ss, device):
    pred = torch.tensor([[[[1.0, 10.0], [10.0, 20.0]]]], device=device)
    gt = torch.tensor([[[[0.0, 10.0], [10.0, 20.0]]]], device=device)
    mae, mse = ss.compute_depth_mae_mse(pred, gt, valid_min=0.0, valid_max=150.0)
    assert mae.item() == pytest.approx(0.0)
    assert mse.item() == pytest.approx(0.0)


def test_get_pointmap_from_depth_z_forward(ss, device):
    b, v, h, w = 1, 1, 8, 8
    depth = torch.full((b, v, h, w), 2.0, device=device)
    k = torch.eye(3, device=device).view(1, 1, 3, 3).expand(b, v, 3, 3).clone()
    k[:, :, 0, 0] = 100.0
    k[:, :, 1, 1] = 100.0
    k[:, :, 0, 2] = w / 2
    k[:, :, 1, 2] = h / 2
    c2w = torch.eye(4, device=device).view(1, 1, 4, 4).expand(b, v, 4, 4).clone()
    pts = ss.get_pointmap_from_depth(depth, k, c2w)
    cx, cy = w // 2, h // 2
    assert pts.shape == (b, v, h, w, 3)
    assert pts[0, 0, cy, cx, 2].item() == pytest.approx(2.0, rel=1e-4)


def test_interleave_left_right_rgb(ss, device):
    # layout: [L0,L1,L2,L3, first_L, first_R] -> interleaved middle + tail
    x = torch.arange(72, device=device, dtype=torch.float32).view(1, 6, 3, 2, 2)
    y = ss.interleave_left_right(x)
    assert y.shape == x.shape
    assert torch.allclose(y[:, -2:], x[:, -2:])
    assert not torch.equal(y[:, :-2], x[:, :-2])


def test_interleave_left_right_depth(ss, device):
    x = torch.arange(24, device=device, dtype=torch.float32).view(1, 6, 2, 2)
    y = ss.interleave_left_right_depth(x)
    assert y.shape == x.shape


def test_add_local_pitch_is_local_rotation(ss, device):
    c2w = torch.eye(4, device=device)
    pitched = ss.add_local_pitch(c2w, deg=10.0)
    assert pitched.shape == (4, 4)
    assert not torch.allclose(pitched, c2w)
    assert pitched[3, 3].item() == pytest.approx(1.0)
