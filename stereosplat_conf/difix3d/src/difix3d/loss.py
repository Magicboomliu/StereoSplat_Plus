import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.ssim_torch import ssim


def restoration_losses(x_pred: torch.Tensor, x_gt: torch.Tensor, args):
    """
    图像复原：L2（[-1,1] MSE → PSNR）+ (1 - SSIM)（[0,1]）。
    """
    xp = x_pred.float()
    xt = x_gt.float()

    mse = F.mse_loss(xp, xt, reduction="mean")
    loss_l2 = args.lambda_l2 * mse

    xp01 = torch.clamp(xp * 0.5 + 0.5, 0.0, 1.0)
    xt01 = torch.clamp(xt * 0.5 + 0.5, 0.0, 1.0)
    ssim_metric = ssim(xp01, xt01, data_range=1.0, size_average=True)
    loss_ssim = args.lambda_ssim * (1.0 - ssim_metric)

    total = loss_l2 + loss_ssim
    return {
        "loss": total,
        "mse": mse,
        "loss_l2": loss_l2,
        "loss_ssim": loss_ssim,
        "ssim_metric": ssim_metric,
    }
