import numpy as np

def psnr_neg1_to_1(mse: float, eps: float = 1e-10) -> float:
    """PSNR for images in [-1, 1]（动态范围 2）。"""
    return float(10.0 * np.log10(4.0 / (mse + eps)))