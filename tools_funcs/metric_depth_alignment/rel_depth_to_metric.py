import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import os
import sys
import skimage.io
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm



def load_the_depthanytingV2_results(path,scale=50):
    # Read the image in unchanged mode (preserves uint16 format)
    img = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
    img = img/scale
    return img

def load_the_Metric3DV2_results(path,scale=256):
    # Read the image in unchanged mode (preserves uint16 format)
    img = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
    img = img/scale
    return img

def loaded_projected_sparse_depth_and_valid_mask(path):
    depth = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED)).astype(np.float32)/256

    mask1 = depth>0
    mask2 = depth<=80
    
    mask1 = mask1.astype(np.float32)
    mask2 = mask2.astype(np.float32)
    
    mask = mask1 * mask2
    
    return depth, mask
        
def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred

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

def depth2metric(rel_depth,gt_sparse_depth,valid_mask,align_type='LS'):
    '''
    rel_depth: realtive depeth
    gt_sparse_depth: projected sparse depth
    valid_mask: valid regions contains project lidar
    align_type: alignment manner, selected from the "LS" (Linear Square), "Med"(Median Matching)
    ------------------------------------------------------
    reference: 
    (1) "LS" implementation from Marigold(CVPR2024): https://github.com/prs-eth/Marigold/blob/main/src/util/alignment.py#L8
    (2) "Med" implementation from hierarchical-3d-gaussians(SIGGRAPH2024): https://github.com/graphdeco-inria/hierarchical-3d-gaussians/blob/main/preprocess/make_depth_scale.py#L19
    '''
    
    
    if align_type =="LS":
        aligned_pred, scale, shift = align_depth_least_square(gt_arr=gt_sparse_depth,
                                pred_arr=rel_depth,
                                valid_mask_arr = valid_mask.astype(np.bool_)
                                )
    
    elif align_type =="Med":
        aligned_pred, scale, shift = Med_Scaling_Depth(rel_depth,gt_sparse_depth,valid_mask.astype(np.bool_))
    
    elif align_type =="None":
        aligned_pred = rel_depth
        scale =1
        shift =1
    
    else:
        raise NotImplementedError
    
    return aligned_pred, scale, shift 

def normalized_depth(disp):
    disp_range = np.minimum(disp.max() / (disp.min() + 0.001), 50.0)
    disp_max = disp.max()
    disp_min = disp_max / disp_range
    depth = 1 / np.maximum(disp, disp_min)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) # range from 0 ~1
    
    return depth

def Med_Scaling_Depth(rel_depth,gt_sparse_depth,valid_mask):


    gt = gt_sparse_depth.squeeze()  # [H, W]
    pred = rel_depth.squeeze()
    valid_mask = valid_mask.squeeze()


    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))


    t_colmap = np.median(gt_masked )
    s_colmap = np.mean(np.abs(gt_masked  - t_colmap))

    t_mono = np.median(pred_masked)
    s_mono = np.mean(np.abs(pred_masked - t_mono))

    scale = s_colmap / s_mono if s_mono > 1e-6 else 0.0
    offset = t_colmap - t_mono * scale
    
    
    aligned_depth = rel_depth * scale + offset
    return aligned_depth,scale,offset   








if __name__=="__main__":
    
    dpt_type = "Metric3DV2" # select from "depthanythingV2" and "Metric3DV2"
    matched_type = "None" # select from "LS" "Med", "None"
    if dpt_type =="depthanythingV2":
        est_depth_rel_root_path = "/data1/StereoDatasets/KITTI/KITTI360/monocular_depth/monodepthV2/"
    elif dpt_type =='Metric3DV2':
        est_depth_rel_root_path = "/data1/StereoDatasets/KITTI/KITTI360/monocular_depth/Metric3DV2/"
    
    
    
    sparse_gt_projected_root_path = "/data1/StereoDatasets/KITTI/KITTI360/projected_sparse_lidar/"
    target_sequence_name = "2013_05_28_drive_0000_sync"
    image_folder_left = os.path.join(est_depth_rel_root_path,"data_2d_raw",target_sequence_name,"image_00/data_rect/")
    mean_mae = 0
    mean_mse = 0
    
    idx = 0
    for fname in tqdm(sorted(os.listdir(image_folder_left))):
        if "conf" in fname:
            continue
        
        idx = idx +1
        pseudo_depth_path = os.path.join(image_folder_left,fname)
        
        if dpt_type == "depthanythingV2":
            depthanythingV2_est_depth_data = load_the_depthanytingV2_results(path=pseudo_depth_path)    
            normalized_depthanythingV2_depth = normalized_depth(depthanythingV2_est_depth_data)
            gt_sparse_project_depth = pseudo_depth_path.replace(est_depth_rel_root_path,sparse_gt_projected_root_path)
        elif dpt_type == "Metric3DV2":
            metric3d_results = load_the_Metric3DV2_results(path=pseudo_depth_path)
            normalized_depthanythingV2_depth = metric3d_results
            gt_sparse_project_depth = pseudo_depth_path.replace(est_depth_rel_root_path,sparse_gt_projected_root_path).replace("_dpt","")


        assert os.path.exists(gt_sparse_project_depth)

        sparse_depth,valid_mask =loaded_projected_sparse_depth_and_valid_mask(gt_sparse_project_depth)
        
        assert sparse_depth.shape == normalized_depthanythingV2_depth.shape
        assert sparse_depth.shape == valid_mask.shape

        aligned_pred, scale, shift  = depth2metric(rel_depth=normalized_depthanythingV2_depth,
                    gt_sparse_depth=sparse_depth,
                    valid_mask=valid_mask,align_type=matched_type)
        
        mse, mae = compute_depth_errors(gt_depth=sparse_depth, aligned_depth=aligned_pred, valid_mask=valid_mask.astype(np.bool_))
    

        mean_mae+=mae
        mean_mse+=mse
    
    
    mean_mae = mean_mae/idx
    mean_mse = mean_mse/idx
    
    print("mean mse:",mean_mse)
    print("mean mae:",mean_mae)
