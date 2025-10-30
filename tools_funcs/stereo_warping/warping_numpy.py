import numpy as np

def normalize_coords_np(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.shape[1] == 2
    h, w = grid.shape[2:]
    grid = grid.copy()  # Create a copy to avoid modifying the original
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :] / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :] / (h - 1)) - 1  # y: [-1, 1]
    grid = np.transpose(grid, (0, 2, 3, 1))  # [B, H, W, 2]
    return grid

def meshgrid_np(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.shape

    x_range = np.arange(0, w).reshape(1, 1, w).repeat(h, axis=1)  # [1, H, W]
    y_range = np.arange(0, h).reshape(1, h, 1).repeat(w, axis=2)  # [1, H, W]

    grid = np.concatenate((x_range, y_range), axis=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = np.expand_dims(grid, axis=0).repeat(b, axis=0)  # [B, 2, H, W]

    if homogeneous:
        ones = np.ones_like(x_range).reshape(1, 1, h, w).repeat(b, axis=0)  # [B, 1, H, W]
        grid = np.concatenate((grid, ones), axis=1)  # [B, 3, H, W]
        assert grid.shape[1] == 3
    return grid

def bilinear_interpolate_np(im, x, y):
    """Perform bilinear interpolation on an image
    Args:
        im: [B, C, H, W]
        x: [B, H, W] coordinates in range [-1, 1]
        y: [B, H, W] coordinates in range [-1, 1]
    Returns:
        interpolated: [B, C, H, W]
    """
    # Convert from [-1,1] to [0, W-1/H-1]
    x = (x + 1) * (im.shape[3] - 1) / 2
    y = (y + 1) * (im.shape[2] - 1) / 2
    
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    
    # Clip to image boundaries
    x0 = np.clip(x0, 0, im.shape[3]-1)
    x1 = np.clip(x1, 0, im.shape[3]-1)
    y0 = np.clip(y0, 0, im.shape[2]-1)
    y1 = np.clip(y1, 0, im.shape[2]-1)
    
    # Get pixel values
    Ia = im[:, :, y0, x0]
    Ib = im[:, :, y1, x0]
    Ic = im[:, :, y0, x1]
    Id = im[:, :, y1, x1]
    
    # Calculate weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    
    # Add batch dimension to weights
    wa = np.expand_dims(wa, axis=1)
    wb = np.expand_dims(wb, axis=1)
    wc = np.expand_dims(wc, axis=1)
    wd = np.expand_dims(wd, axis=1)
    
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def disp_warp_np(img, disp, padding_mode='border'):
    """Warping by disparity
    Args:
        img: [B, 3, H, W]
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]
    """
    assert np.min(disp) >= 0

    grid = meshgrid_np(img)  # [B, 2, H, W] in image scale
    # Note that -disp here
    offset = np.concatenate((-disp, np.zeros_like(disp)), axis=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords_np(sample_grid)  # [B, H, W, 2] in [-1, 1]
    
    # Split sample_grid into x and y components
    x_coords = sample_grid[:, :, :, 0]
    y_coords = sample_grid[:, :, :, 1]
    
    # Perform bilinear interpolation
    warped_img = bilinear_interpolate_np(img, x_coords, y_coords)
    
    # Create valid mask (1 where interpolation was valid, 0 otherwise)
    mask = np.ones_like(img)
    valid_mask = bilinear_interpolate_np(mask, x_coords, y_coords)
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    
    return warped_img, valid_mask