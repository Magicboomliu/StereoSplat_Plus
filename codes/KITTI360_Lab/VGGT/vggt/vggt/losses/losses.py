import torch
import torch.nn.functional as F

def camera_loss(pred_camera, gt_camera, delta=1.0):
    """
    pred_camera, gt_camera: shape [N, 7] (or [B, 7])
    """
    return F.huber_loss(pred_camera, gt_camera, delta=delta, reduction='mean')


def gradient_map(depth):
    # depth: [B, V, H, W] (4D tensor)
    dx = depth[:, :, :, :-1] - depth[:, :, :, 1:]  # [B, V, H, W-1]
    dy = depth[:, :, :-1, :] - depth[:, :, 1:, :]  # [B, V, H-1, W]



    dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')       # pad W axis
    dy = F.pad(dy, (0, 0, 0, 1), mode='replicate') # pad H axis

    
    return torch.cat([dx, dy], dim=1)  # [B, 2V, H, W]

def depth_loss(pred_depth, gt_depth, sigma_d, alpha=0.5):
    """
    pred_depth, gt_depth, sigma_d: shape [B, 1, H, W]
    """
    abs_diff = torch.abs(pred_depth - gt_depth)  # [B,1,H,W]
    grad_diff = torch.abs(gradient_map(pred_depth) - gradient_map(gt_depth))  # [B,2,H,W]

    loss = sigma_d * abs_diff + sigma_d * grad_diff.sum(dim=1, keepdim=True) - alpha * torch.log(sigma_d + 1e-6)
    return loss.mean()


def gradient_map_vec3(pcd):
    """
    pcd: [B, 3, H, W]
    返回：每个方向的 gradient [B, 6, H, W] (dx, dy for each channel)
    """
    grads = []
    for c in range(3):
        dx = pcd[:, c:c+1, :, :-1] - pcd[:, c:c+1, :, 1:]
        dy = pcd[:, c:c+1, :-1, :] - pcd[:, c:c+1, 1:, :]
        dx = F.pad(dx, (0,1,0,0), mode='replicate')
        dy = F.pad(dy, (0,0,0,1), mode='replicate')
        grads.extend([dx, dy])
    return torch.cat(grads, dim=1)  # [B, 6, H, W]

def pcd_loss(pred_pcd, gt_pcd, sigma_p, alpha=0.5):
    """
    pred_pcd, gt_pcd: [B, 3, H, W]
    sigma_p: [B, 1, H, W] or [B, 3, H, W] → broadcastable
    """
    diff = torch.norm(pred_pcd - gt_pcd, dim=1, keepdim=True)  # [B,1,H,W]
    grad_pred = gradient_map_vec3(pred_pcd)  # [B,6,H,W]
    grad_gt = gradient_map_vec3(gt_pcd)
    grad_diff = torch.abs(grad_pred - grad_gt).sum(dim=1, keepdim=True)  # [B,1,H,W]

    loss = sigma_p * diff + sigma_p * grad_diff - alpha * torch.log(sigma_p + 1e-6)
    return loss.mean()

