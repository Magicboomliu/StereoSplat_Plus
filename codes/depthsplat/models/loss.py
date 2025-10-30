import torch
import torch.nn.functional as F

def compute_gradient_x(img):
    return img[..., :-1] - img[..., 1:]

def compute_gradient_y(img):
    return img[..., :-1, :] - img[..., 1:, :]

def depth_loss(depth_pred, depth_gt, alpha=20, beta=20):
    """
    depth_pred, depth_gt: tensors of shape [B, V, H, W]
    """
    assert depth_pred.shape == depth_gt.shape, "Shape mismatch between prediction and ground truth"

    # Create mask: only consider depth values between 0 and 150
    mask = (depth_gt > 0) & (depth_gt < 200)

    # Basic L1 loss with mask
    l1_loss = torch.abs(depth_pred - depth_gt)
    l1_loss = l1_loss * mask  # apply mask

    # Compute gradients
    grad_x_pred = compute_gradient_x(depth_pred)
    grad_x_gt = compute_gradient_x(depth_gt.detach())
    grad_y_pred = compute_gradient_y(depth_pred)
    grad_y_gt = compute_gradient_y(depth_gt.detach())

    # Gradient L1 loss
    grad_loss_x = torch.abs(grad_x_pred - grad_x_gt)
    grad_loss_y = torch.abs(grad_y_pred - grad_y_gt)

    grad_loss_x = F.pad(grad_loss_x, (0,1), mode='replicate')
    grad_loss_y = F.pad(grad_loss_y, (0,0,0,1), mode='replicate')
    # grad_x_mask = F.pad(grad_x_mask, (0,1), mode='replicate')
    # grad_y_mask = F.pad(grad_y_mask, (0,0,0,1), mode='replicate')

    # Apply gradient masks
    grad_loss_x = grad_loss_x 
    grad_loss_y = grad_loss_y 

    total_loss = alpha * l1_loss + beta * (grad_loss_x + grad_loss_y)

    # Average only over valid mask locations
    valid_mask = mask
    return total_loss.sum() / (valid_mask.sum().float() + 1e-8)

def depth_l1_loss(depth_pred, depth_gt):
    
    mask = (depth_gt > 0) & (depth_gt < 200)
    mask = depth_gt.float()
    """
    Compute L1 loss between predicted and ground truth depth maps.
    
    Args:
        depth_pred (torch.Tensor): Predicted depth, shape [B, V, H, W]
        depth_gt (torch.Tensor): Ground truth depth, shape [B, V, H, W]
    
    Returns:
        torch.Tensor: Scalar L1 loss value
    """
    assert depth_pred.shape == depth_gt.shape, "Shape mismatch between prediction and ground truth"
    
    return torch.sum(torch.abs(depth_pred*mask - depth_gt*mask))/(mask.sum()+1e-6)