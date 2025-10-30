import torch
import torch.nn.functional as F
import math
import numpy as np

def compute_rra_rta_absolute(est: torch.Tensor, gt: torch.Tensor):
    """
    est, gt: [B, V, 4, 4] 的相机位姿矩阵（cam2world）
    返回:
        rra: [B, V] 相对旋转误差 (deg)
        rta: [B, V] 相对平移方向误差 (deg)
    """
    assert est.shape == gt.shape
    B, V, _, _ = est.shape

    # 取旋转和平移
    R_est = est[..., :3, :3]   # [B, V, 3, 3]
    T_est = est[..., :3, 3]    # [B, V, 3]

    R_gt = gt[..., :3, :3]     # [B, V, 3, 3]
    T_gt = gt[..., :3, 3]      # [B, V, 3]

    # === Rotation error ===
    R_rel = torch.matmul(R_est.transpose(-2, -1), R_gt)  # R_est^T * R_gt
    trace = R_rel.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)  # [B, V]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    rra = torch.acos(cos_theta) * (180.0 / math.pi)  # [B, V]

    # === Translation direction error ===
    t_est_norm = F.normalize(T_est, dim=-1)
    t_gt_norm = F.normalize(T_gt, dim=-1)
    dot = (t_est_norm * t_gt_norm).sum(-1)  # [B, V]
    dot = torch.clamp(dot, -1.0, 1.0)
    rta = torch.acos(dot) * (180.0 / math.pi)  # [B, V]
    
    
    rra = rra.mean()
    rta = rta.mean()
    
    return rra, rta  # shape: [B, V]

def compute_depth_mae_mse(depth_pred, depth_gt, valid_min=0.0, valid_max=150.0):
    """
    Computes MAE and MSE between predicted and GT depth maps, with optional valid range filtering.

    Args:
        depth_pred (torch.Tensor): Predicted depth, shape [B, V, H, W]
        depth_gt (torch.Tensor): Ground truth depth, shape [B, V, H, W]
        valid_min (float): minimum valid GT depth
        valid_max (float): maximum valid GT depth

    Returns:
        mae (torch.Tensor): scalar mean absolute error
        mse (torch.Tensor): scalar mean squared error
    """
    assert depth_pred.shape == depth_gt.shape, "Shape mismatch between prediction and GT"

    # Create valid mask (only use pixels with valid GT depth)
    valid_mask = (depth_gt > valid_min) & (depth_gt < valid_max)

    # Compute errors
    abs_error = torch.abs(depth_pred - depth_gt)
    sq_error = (depth_pred - depth_gt) ** 2

    # Apply mask
    abs_error = abs_error[valid_mask]
    sq_error = sq_error[valid_mask]

    # Final metrics
    mae = abs_error.mean()
    mse = sq_error.mean()

    return mae, mse

def compute_pcd_mae_mse(est_pcd, gt_pcd):
    """
    Computes MAE and MSE between predicted and GT depth maps, with optional valid range filtering.

    Args:
        depth_pred (torch.Tensor): Predicted depth, shape [B, V, H, W]
        depth_gt (torch.Tensor): Ground truth depth, shape [B, V, H, W]
        valid_min (float): minimum valid GT depth
        valid_max (float): maximum valid GT depth

    Returns:
        mae (torch.Tensor): scalar mean absolute error
        mse (torch.Tensor): scalar mean squared error
    """
    assert est_pcd.shape == gt_pcd.shape, "Shape mismatch between prediction and GT"

    # Compute errors
    abs_error = torch.abs(est_pcd - gt_pcd)
    sq_error = (est_pcd - gt_pcd) ** 2

    # Apply mask
    abs_error = abs_error
    sq_error = sq_error

    # Final metrics
    mae = abs_error.mean()
    mse = sq_error.mean()

    return mae, mse

def convert_depth_to_disp(factor=328.318735,depth=None):
    
    mask = depth>0
    mask = mask.astype(np.float32)

    disparity = factor / (depth +1e-3)
    disparity = disparity * mask
    disparity = np.clip(disparity,a_max=220,a_min=0)
    
    disparity = kitti_colormap(disparity)
    return disparity

def kitti_colormap(disparity, maxval=-1):
	"""
	A utility function to reproduce KITTI fake colormap
	Arguments:
	  - disparity: numpy float32 array of dimension HxW
	  - maxval: maximum disparity value for normalization (if equal to -1, the maximum value in disparity will be used)
	
	Returns a numpy uint8 array of shape HxWx3.
	"""
	if maxval < 0:
		maxval = np.max(disparity)

	colormap = np.asarray([[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],[0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0]])
	weights = np.asarray([8.771929824561404,5.405405405405405,8.771929824561404,5.747126436781609,8.771929824561404,5.405405405405405,8.771929824561404,0])
	cumsum = np.asarray([0,0.114,0.299,0.413,0.587,0.701,0.8859999999999999,0.9999999999999999])

	colored_disp = np.zeros([disparity.shape[0], disparity.shape[1], 3])
	values = np.expand_dims(np.minimum(np.maximum(disparity/maxval, 0.), 1.), -1)
	bins = np.repeat(np.repeat(np.expand_dims(np.expand_dims(cumsum,axis=0),axis=0), disparity.shape[1], axis=1), disparity.shape[0], axis=0)
	diffs = np.where((np.repeat(values, 8, axis=-1) - bins) > 0, -1000, (np.repeat(values, 8, axis=-1) - bins))
	index = np.argmax(diffs, axis=-1)-1

	w = 1-(values[:,:,0]-cumsum[index])*np.asarray(weights)[index]


	colored_disp[:,:,2] = (w*colormap[index][:,:,0] + (1.-w)*colormap[index+1][:,:,0])
	colored_disp[:,:,1] = (w*colormap[index][:,:,1] + (1.-w)*colormap[index+1][:,:,1])
	colored_disp[:,:,0] = (w*colormap[index][:,:,2] + (1.-w)*colormap[index+1][:,:,2])

	return (colored_disp*np.expand_dims((disparity>0),-1)*255).astype(np.uint8)



class Pose_Quality_Meter(object):
    def __init__(self,rra,rta):
        self.rra = rra
        self.rta = rta
        self.counter = 0
        
    def update(self,rra,rta):
        self.rra +=rra
        self.rta +=rta
        self.counter = self.counter +1
        
    def get_stats(self):
        if self.counter==0:
            return {"rra":0,
                    "rta":0}
        else:
            return {
                "rra": self.rra * 1.0 / self.counter,
                "rta": self.rta * 1.0 /self.counter
            }


class Depth_Quality_Meter(object):
    def __init__(self,mae,mse):
        self.mae = mae
        self.mse = mse
        self.counter =0
    
    def update(self,mae,mse):
        self.mae +=mae
        self.mse +=mse
        self.counter +=1
        
    def get_stats(self):
        if self.counter==0:
            return {"mae":0,
                    "mse":0}
        else:
            return {
                "mae": self.mae * 1.0 / self.counter,
                "mse": self.mse * 1.0 /self.counter
            }


class Pcd_Quality_Meter(object):
    def __init__(self,mae,mse):
        self.mae = mae
        self.mse = mse
        self.counter =0
    
    def update(self,mae,mse):
        self.mae +=mae
        self.mse +=mse
        self.counter +=1
        
    def get_stats(self):
        if self.counter==0:
            return {"mae":0,
                    "mse":0}
        else:
            return {
                "mae": self.mae * 1.0 / self.counter,
                "mse": self.mse * 1.0 /self.counter
            }
