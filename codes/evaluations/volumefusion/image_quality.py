import torch
import torch.nn.functional as F
from math import log10

# --- PSNR ---
def psnr(img_est, img_gt, max_val=1.0):
    mse = F.mse_loss(img_est, img_gt, reduction='mean')
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)

# --- SSIM (per-frame averaged) ---
# Reference: https://ece.uwaterloo.ca/~z70wang/research/ssim/
def ssim(img_est, img_gt, max_val=1.0, window_size=11, C1=None, C2=None):
    # assume input [1,V,3,H,W], process frame by frame
    B, V, C, H, W = img_est.shape
    img_est = img_est.view(B*V, C, H, W)
    img_gt  = img_gt.view(B*V, C, H, W)

    # Gaussian window (approx)
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords**2) / (2*sigma**2))
        g /= g.sum()
        return g

    # 1D → 2D Gaussian kernel
    window = gaussian_window(window_size).unsqueeze(1)
    window = window @ window.T
    window = window / window.sum()
    window = window.view(1,1,window_size,window_size).to(img_est.device)

    if C1 is None:
        C1 = (0.01 * max_val) ** 2
    if C2 is None:
        C2 = (0.03 * max_val) ** 2

    mu1 = F.conv2d(img_est, window.expand(C,1,window_size,window_size), groups=C, padding=window_size//2)
    mu2 = F.conv2d(img_gt,  window.expand(C,1,window_size,window_size), groups=C, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img_est * img_est, window.expand(C,1,window_size,window_size), groups=C, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img_gt  * img_gt, window.expand(C,1,window_size,window_size), groups=C, padding=window_size//2) - mu2_sq
    sigma12   = F.conv2d(img_est * img_gt,  window.expand(C,1,window_size,window_size), groups=C, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# Example
img_est = torch.rand(1, 5, 3, 128, 128)  # dummy est
img_gt  = torch.rand(1, 5, 3, 128, 128)  # dummy gt

psnr_val = psnr(img_est, img_gt)
ssim_val = ssim(img_est, img_gt)

print("PSNR:", psnr_val.item())
print("SSIM:", ssim_val.item())
