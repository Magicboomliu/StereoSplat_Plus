import os
import math
import numpy as np
from PIL import Image


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _to_uint8_gray(x):
    """
    x: [H, W], float or bool
    return: uint8 [H, W]
    """
    if x.dtype == np.bool_:
        return x.astype(np.uint8) * 255

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _normalize_to_uint8(x, vmin=None, vmax=None, log_scale=False):
    """
    x: [H, W], float
    normalize to [0,255] for visualization
    """
    x = np.array(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if log_scale:
        x = np.log1p(np.maximum(x, 0.0))

    if vmin is None:
        vmin = float(np.min(x))
    if vmax is None:
        vmax = float(np.max(x))

    if vmax - vmin < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)

    x = (x - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _save_gray_image(arr, save_path):
    """
    arr: [H, W], uint8
    """
    Image.fromarray(arr, mode="L").save(save_path)


def _make_grid(images, nrow=None, pad=2, pad_value=0):
    """
    images: list of [H, W] uint8 arrays
    return: one big [H_grid, W_grid] uint8 array
    """
    assert len(images) > 0

    h, w = images[0].shape
    n = len(images)

    if nrow is None:
        nrow = int(math.ceil(math.sqrt(n)))
    ncol = int(math.ceil(n / nrow))

    grid_h = ncol * h + (ncol - 1) * pad
    grid_w = nrow * w + (nrow - 1) * pad
    grid = np.full((grid_h, grid_w), pad_value, dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // nrow
        c = idx % nrow
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        grid[y0:y0+h, x0:x0+w] = img

    return grid


def save_debug_visualizations(
    debug,
    save_root,
    batch_names=None,
    save_single_views=True,
    save_grids=True,
    depth_log_scale=False,
    rel_err_max=0.5,
):
    """
    Save debug tensors from fuse_rgb_by_projected_depth_v2 as images.

    Args:
        debug: dict returned by fuse_rgb_by_projected_depth_v2(..., return_debug=True)
               expected keys:
                   - projected_depth      [B,V,H,W]
                   - projected_valid      [B,V,H,W]
                   - rel_err_proj_plus    [B,V,H,W]
                   - rel_err_proj_base    [B,V,H,W]
                   - consistent_mask      [B,V,H,W]
                   - plus_front_mask      [B,V,H,W]
                   - base_front_mask      [B,V,H,W]
        save_root: output directory
        batch_names: optional list[str] of length B
        save_single_views: save per-view images
        save_grids: save one grid image per key per batch
        depth_log_scale: whether to visualize depth with log1p
        rel_err_max: max value for relative error visualization
    """
    _ensure_dir(save_root)

    required_keys = [
        "projected_depth",
        "projected_valid",
        "rel_err_proj_plus",
        "rel_err_proj_base",
        "consistent_mask",
        "plus_front_mask",
        "base_front_mask",
    ]
    for k in required_keys:
        if k not in debug:
            raise KeyError(f"Missing key in debug: {k}")

    proj_depth = debug["projected_depth"].detach().float().cpu().numpy()
    proj_valid = debug["projected_valid"].detach().float().cpu().numpy()
    rel_err_pp = debug["rel_err_proj_plus"].detach().float().cpu().numpy()
    rel_err_pb = debug["rel_err_proj_base"].detach().float().cpu().numpy()
    consistent = debug["consistent_mask"].detach().float().cpu().numpy()
    plus_front = debug["plus_front_mask"].detach().float().cpu().numpy()
    base_front = debug["base_front_mask"].detach().float().cpu().numpy()

    B, V, H, W = proj_depth.shape

    if batch_names is None:
        batch_names = [f"batch_{b:03d}" for b in range(B)]
    if len(batch_names) != B:
        raise ValueError(f"batch_names length should be {B}, got {len(batch_names)}")

    for b in range(B):
        batch_dir = os.path.join(save_root, batch_names[b])
        _ensure_dir(batch_dir)

        grid_buffers = {
            "projected_depth": [],
            "projected_valid": [],
            "rel_err_proj_plus": [],
            "rel_err_proj_base": [],
            "consistent_mask": [],
            "plus_front_mask": [],
            "base_front_mask": [],
        }

        # 为 projected_depth 做统一范围，只在 valid 区域统计
        valid_mask_b = proj_valid[b] > 0.5
        if np.any(valid_mask_b):
            valid_depth_vals = proj_depth[b][valid_mask_b]
            depth_vmin = float(np.min(valid_depth_vals))
            depth_vmax = float(np.max(valid_depth_vals))
        else:
            depth_vmin, depth_vmax = 0.0, 1.0

        for v in range(V):
            # projected depth
            depth_img = _normalize_to_uint8(
                proj_depth[b, v],
                vmin=depth_vmin,
                vmax=depth_vmax,
                log_scale=depth_log_scale,
            )

            # invalid 区域置 0，视觉上更清楚
            depth_img = depth_img.copy()
            depth_img[proj_valid[b, v] < 0.5] = 0

            valid_img = _to_uint8_gray(proj_valid[b, v] > 0.5)
            consistent_img = _to_uint8_gray(consistent[b, v] > 0.5)
            plus_front_img = _to_uint8_gray(plus_front[b, v] > 0.5)
            base_front_img = _to_uint8_gray(base_front[b, v] > 0.5)

            rel_err_pp_img = _normalize_to_uint8(
                np.clip(rel_err_pp[b, v], 0.0, rel_err_max),
                vmin=0.0,
                vmax=rel_err_max,
                log_scale=False,
            )
            rel_err_pb_img = _normalize_to_uint8(
                np.clip(rel_err_pb[b, v], 0.0, rel_err_max),
                vmin=0.0,
                vmax=rel_err_max,
                log_scale=False,
            )

            grid_buffers["projected_depth"].append(depth_img)
            grid_buffers["projected_valid"].append(valid_img)
            grid_buffers["rel_err_proj_plus"].append(rel_err_pp_img)
            grid_buffers["rel_err_proj_base"].append(rel_err_pb_img)
            grid_buffers["consistent_mask"].append(consistent_img)
            grid_buffers["plus_front_mask"].append(plus_front_img)
            grid_buffers["base_front_mask"].append(base_front_img)

            if save_single_views:
                _save_gray_image(depth_img, os.path.join(batch_dir, f"v{v:02d}_projected_depth.png"))
                _save_gray_image(valid_img, os.path.join(batch_dir, f"v{v:02d}_projected_valid.png"))
                _save_gray_image(rel_err_pp_img, os.path.join(batch_dir, f"v{v:02d}_rel_err_proj_plus.png"))
                _save_gray_image(rel_err_pb_img, os.path.join(batch_dir, f"v{v:02d}_rel_err_proj_base.png"))
                _save_gray_image(consistent_img, os.path.join(batch_dir, f"v{v:02d}_consistent_mask.png"))
                _save_gray_image(plus_front_img, os.path.join(batch_dir, f"v{v:02d}_plus_front_mask.png"))
                _save_gray_image(base_front_img, os.path.join(batch_dir, f"v{v:02d}_base_front_mask.png"))

        if save_grids:
            for key, imgs in grid_buffers.items():
                grid = _make_grid(imgs, nrow=None, pad=2, pad_value=0)
                _save_gray_image(grid, os.path.join(batch_dir, f"{key}_grid.png"))