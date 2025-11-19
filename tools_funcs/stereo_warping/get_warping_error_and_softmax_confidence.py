import os
from warping_numpy import disp_warp_np
from warping_torch import disp_warp
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def load_the_Metric3DV2_Results(path,scale=256):
    # Read the image in unchanged mode (preserves uint16 format)
    img = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
    img = img/scale
    return img

def loaded_image(path):
    results = np.array(Image.open(path).convert("RGB")).astype(np.float32)
    return results
    

def convert_depth_into_disp(depth,focal_length,baseline,eps=1e-6,largest=320):
    
    disp = focal_length * baseline/(depth+1e-6)
    disp = np.clip(disp,a_min=0, a_max=largest)
    return disp

def get_conf_and_warped_error(left_image_data,right_image_data,metric_depth,focal_length=552.554261,baseline=0.5941836995443964,
                              temp=200):
  
    if np.max(left_image_data) and np.max(right_image_data)>2:
        temp = temp
    else:
        temp = 1
    
    disp = convert_depth_into_disp(metric_depth,focal_length,baseline,eps=1e-6,largest=320)
    

    left_image_data_t = torch.from_numpy(left_image_data).permute(2,0,1).unsqueeze(0)
    
    right_image_data_t = torch.from_numpy(right_image_data).permute(2,0,1).unsqueeze(0)
    
    disp = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0)

    warped_left, valid_mask_left = disp_warp(img=right_image_data_t, disp=disp, padding_mode='border')
    
    error = torch.abs(warped_left-left_image_data_t)
    error = torch.sum(error,dim=1,keepdim=True)
    
    conf =torch.exp(-error/temp)
    
    error = error.squeeze(0).squeeze(0).cpu().numpy()
    conf = conf.squeeze(0).squeeze(0).cpu().numpy()

    return error, conf



def loaded_projected_sparse_depth_and_valid_mask(path):
    depth = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED)).astype(np.float32)/256

    mask1 = depth>0
    mask2 = depth<=80
    
    mask1 = mask1.astype(np.float32)
    mask2 = mask2.astype(np.float32)
    
    mask = mask1 * mask2
    
    return depth, mask


def compute_depth_errors(gt_depth: np.ndarray, aligned_depth: np.ndarray, valid_mask: np.ndarray):
    """
    Compute MSE and MAE between GT and aligned prediction in valid regions.

    Args:
        gt_depth (np.ndarray): Ground truth depth map, shape [H, W] or [N].
        aligned_depth (np.ndarray): Aligned predicted depth map, same shape as gt_depth.
        valid_mask (np.ndarray): Boolean mask of valid pixels, same shape.

    Returns:
        mse (float): Mean squared error.
        mae (float): Mean absolute error.
    """
    gt_valid = gt_depth[valid_mask]
    pred_valid = aligned_depth[valid_mask]

    mse = np.mean((gt_valid - pred_valid) ** 2)
    mae = np.mean(np.abs(gt_valid - pred_valid))

    return mse, mae

if __name__=="__main__":
    
    left_image = "/media/zliu/data12/dataset/KITTI/KITTI360/image_data/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000259.png"
    right_image = left_image.replace("image_00","image_01")
    metric_depth = "/media/zliu/data12/dataset/KITTI/KITTI360/PseudoDepth_NMRFStereo/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000259.png"
    
    gt_depth = "/media/zliu/data12/dataset/KITTI/VSRD_Format/projected_sparse_lidar/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000259.png"
    
    assert os.path.exists(left_image)
    assert os.path.exists(right_image)
    assert os.path.exists(metric_depth)
    
    
    left_image_data = loaded_image(left_image)
    right_image_data = loaded_image(right_image)
    metric_depth = load_the_Metric3DV2_Results(metric_depth)
    
    gt_depth_sparse,gt_mask = loaded_projected_sparse_depth_and_valid_mask(gt_depth)
    
    mse, mae = compute_depth_errors(gt_depth=gt_depth_sparse,aligned_depth=metric_depth,
                         valid_mask=gt_mask.astype(np.bool_))
    


    error,conf= get_conf_and_warped_error(left_image_data,right_image_data,metric_depth,focal_length=552.554261,baseline=0.5941836995443964,
                              temp=200)
    
    
        


    
        
    # pass