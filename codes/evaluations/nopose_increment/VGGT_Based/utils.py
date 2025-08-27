import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import numpy as np

def select_confident_points(points: torch.Tensor, conf: torch.Tensor, conf_thresh: float = 0.5):
    """
    Args:
        points: (1, H, W, 3) 点云坐标
        conf:   (1, H, W) 置信度
        conf_thresh: float，阈值，保留大于该值的点

    Returns:
        selected_points: (N, 3)
    """
    # 变成 (H, W)
    conf = conf.squeeze(0)          # (H, W)
    points = points.squeeze(0)      # (H, W, 3)
    # 置信度mask
    mask = conf > conf_thresh       # (H, W)
    # 选点
    selected_points = points[mask]  # (N, 3)

    return selected_points



import torch
from typing import Tuple

def img2cam_sparse_torch(depth_map: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    将稀疏深度图转换为相机坐标系下的 3D 点（仅保留有效像素）。

    Args:
        depth_map: (H, W)  稀疏深度图，0 表示无效（float 或 half 等皆可）
        K        : (3, 3)  相机内参（与 depth_map 在同一 device 上）

    Returns:
        cam_points: (N, 3)  相机坐标系下的 3D 点（X, Y, Z）
    """
    assert depth_map.ndim == 2, "depth_map 必须是 (H, W)"
    assert K.shape == (3, 3), "K 必须是 (3, 3)"
    H, W = depth_map.shape
    device = depth_map.device
    dtype  = depth_map.dtype

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid_mask = depth_map > 0
    v, u = torch.nonzero(valid_mask, as_tuple=True)   # v: row, u: col   shape: (N,)
    z = depth_map[v, u]                               # (N,)

    # 反投影到相机坐标：X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy, Z=Z
    x = (u.to(dtype) - cx) * z / fx
    y = (v.to(dtype) - cy) * z / fy

    cam_points = torch.stack([x, y, z], dim=-1)      # (N, 3)
    return cam_points


def cam2image_torch(points: torch.Tensor,
                    K: torch.Tensor,
                    height: int,
                    width: int,
                    depth_range: Tuple[float, float] = (0.0, 100.0)) -> torch.Tensor:
    """
    将相机坐标系的点投影回图像，生成稀疏深度图。

    Args:
        points      : (N, 3)  相机坐标系 3D 点（单位与 K 一致，Z>0 在相机前方）
        K           : (3, 3)  相机内参
        height, width: 输出图像大小
        depth_range : (min_depth, max_depth)，仅保留此范围内的深度

    Returns:
        depth_map: (H, W)  稀疏深度图（无值为 0）
    """
    assert points.ndim == 2 and points.shape[1] == 3, "points 需要是 (N, 3)"
    assert K.shape == (3, 3), "K 必须是 (3, 3)"

    device = points.device
    dtype  = points.dtype

    # 齐次投影 u' = K * X
    # points_T: (3, N)
    points_T = points.t()  # (3, N)
    proj = K.to(device=device, dtype=dtype) @ points_T  # (3, N)

    z = proj[2]  # (N,)
    # 计算像素坐标（四舍五入到整数像素）
    # 避免除 0，先不改 z，只用有效 mask 过滤
    u = torch.round(proj[0] / z).to(torch.int64)  # (N,)
    v = torch.round(proj[1] / z).to(torch.int64)  # (N,)

    # 有效性检查：在图像范围内、深度正且在给定范围内
    min_d, max_d = depth_range
    valid = (
        (u >= 0) & (u < width) &
        (v >= 0) & (v < height) &
        torch.isfinite(z) &
        (z > min_d) & (z < max_d)
    )

    if valid.any():
        u = u[valid]
        v = v[valid]
        z = z[valid]
    else:
        return torch.zeros((height, width), device=device, dtype=dtype)

    # 构建稀疏深度：若多点落同一像素，取最近（z 最小）
    depth_map = torch.full((height, width), float("inf"), device=device, dtype=dtype)
    lin_idx = v * width + u  # (M,)

    # 需要 PyTorch >= 1.12：使用 amin 归约
    depth_map = depth_map.view(-1)
    depth_map.scatter_reduce_(0, lin_idx, z, reduce="amin")
    depth_map = depth_map.view(height, width)

    # 将未写入的位置设为 0
    depth_map[torch.isinf(depth_map)] = 0

    return depth_map


def resize_the_sparse_lidar_torch(depthmap: torch.Tensor,
                                  raw_K: torch.Tensor,
                                  after_K: torch.Tensor,
                                  height: int,
                                  width: int,
                                  depth_range: float = 80) -> torch.Tensor:
    """
    用新的内参/分辨率重映射稀疏深度图。

    Args:
        depthmap : (H0, W0)  原始稀疏深度
        raw_K    : (3, 3)    原始内参
        after_K  : (3, 3)    新的（目标）内参
        height, width:       目标图像大小
        depth_range:         最大深度（最小深度固定为 0）

    Returns:
        resize_gt_sparse_depth: (height, width)  新尺寸/内参下的稀疏深度图
    """
    # 1) (H0, W0) -> (N, 3) 相机系点
    points_cam = img2cam_sparse_torch(depth_map=depthmap, K=raw_K)  # (N, 3)

    # 2) 用新内参投影并成图
    out = cam2image_torch(points=points_cam,
                          K=after_K,
                          height=height,
                          width=width,
                          depth_range=(0.0, float(depth_range)))
    return out

