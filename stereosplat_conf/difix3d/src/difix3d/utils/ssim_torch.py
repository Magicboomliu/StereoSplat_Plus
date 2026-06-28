"""
可微 SSIM（Wang et al.），不依赖 pytorch-msssim。
输入约定与常见实现一致：BxCxHxW，数值已在 [0, data_range]。
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

_WINDOW_CACHE: Dict[Tuple, torch.Tensor] = {}


def _gaussian_1d(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    return g / g.sum()


def _get_window(window_size: int, channel: int, device, dtype) -> torch.Tensor:
    key = (window_size, channel, device, dtype)
    if key not in _WINDOW_CACHE:
        g1d = _gaussian_1d(window_size, 1.5, device, dtype)
        w2d = (g1d[:, None] @ g1d[None, :]).unsqueeze(0).unsqueeze(0)
        win = w2d.expand(channel, 1, window_size, window_size).contiguous()
        _WINDOW_CACHE[key] = win
    return _WINDOW_CACHE[key]


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    size_average: bool = True,
    window_size: int = 11,
) -> torch.Tensor:
    """
    Args:
        x, y: (B, C, H, W), same shape
        data_range: 动态范围（如 [0,1] 则为 1.0）
        size_average: True 返回标量均值；False 返回 per-image SSIM (B,)
    """
    if x.shape != y.shape:
        raise ValueError(f"ssim: shape mismatch {x.shape} vs {y.shape}")
    if x.dim() != 4:
        raise ValueError("ssim: expected (B, C, H, W)")

    b, c, h, w = x.shape
    if min(h, w) < window_size:
        raise ValueError(f"ssim: need H,W >= {window_size}, got {h}x{w}")

    x = x.float()
    y = y.float()
    device, dtype = x.device, x.dtype

    win = _get_window(window_size, c, device, dtype)
    pad = window_size // 2

    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2

    mu_x = F.conv2d(x, win, padding=pad, groups=c)
    mu_y = F.conv2d(y, win, padding=pad, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, win, padding=pad, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, win, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, win, padding=pad, groups=c) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    m = num / (den + 1e-12)

    if size_average:
        return m.mean()
    return m.view(b, -1).mean(dim=1)
