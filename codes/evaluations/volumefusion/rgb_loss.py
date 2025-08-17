import torch
import torch.nn.functional as F

# ---- helpers ----
def _gaussian_kernel(kernel_size=11, sigma=1.5, channels=3, device='cpu', dtype=torch.float32):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1)/2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1)              # [K,1]
    k2d = (g @ g.t())
    k2d = k2d / k2d.sum()
    return k2d.expand(channels, 1, kernel_size, kernel_size).contiguous()  # [C,1,K,K]

def _ssim_map(x, y, kernel, max_val=1.0):
    """
    x,y: [N,C,H,W] in [0, max_val]
    kernel: [C,1,K,K], depthwise
    """
    C = x.shape[1]
    pad = kernel.shape[-1] // 2
    conv = lambda z: F.conv2d(z, kernel, groups=C, padding=pad)

    mu_x, mu_y = conv(x), conv(y)
    mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y

    sigma_x2 = conv(x*x) - mu_x2
    sigma_y2 = conv(y*y) - mu_y2
    sigma_xy = conv(x*y) - mu_xy

    L = float(max_val)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    num = (2*mu_xy + C1) * (2*sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim = (num / (den + 1e-12)).clamp(min=-1.0, max=1.0)
    return ssim

# ---- main loss ----
def rgb_loss_l1_dssim_5d(render, target, alpha=0.2, mask=None,
                         kernel_size=11, sigma=1.5, max_val=1.0, reduction='mean'):
    """
    render/target: [B,V,3,H,W], in [0,1] (set max_val accordingly)
    mask (optional): [B,V,1,H,W] or broadcastable; 1=valid
    return: loss, {'l1':..., 'dssim':...}
    """
    assert render.shape == target.shape and render.dim() == 5 and render.size(2) in (1,3), \
        "Expect [B,V,C,H,W] with C=1 or 3."

    B, V, C, H, W = render.shape
    dev, dt = render.device, render.dtype

    r = render.reshape(B*V, C, H, W)
    t = target.reshape(B*V, C, H, W)

    if mask is not None:
        m = mask.to(device=dev, dtype=dt)
        # broadcast to [B,V,C,H,W] then flatten
        if m.dim() == 5 and m.size(2) == 1 and C > 1:
            m = m.expand(B, V, C, H, W)
        m = m.reshape(B*V, C, H, W)
    else:
        m = None

    # L1
    if m is not None:
        l1 = (m * (r - t).abs()).sum() / (m.sum() + 1e-12)
    else:
        l1 = (r - t).abs().mean()

    # D-SSIM
    kernel = _gaussian_kernel(kernel_size, sigma, channels=C, device=dev, dtype=dt)
    ssim = _ssim_map(r, t, kernel, max_val=max_val)          # [N,C,H,W]
    dssim = (1.0 - ssim) / 2.0
    if m is not None:
        dssim = (dssim * m).sum() / (m.sum() + 1e-12)
    else:
        dssim = dssim.mean()

    loss = (1 - alpha) * l1 + alpha * dssim
    if reduction == 'sum':
        # 注意：‘sum’这里是对标量的同义返回；通常不需要
        pass
    return loss, {'l1': l1.detach(), 'dssim': dssim.detach()}
