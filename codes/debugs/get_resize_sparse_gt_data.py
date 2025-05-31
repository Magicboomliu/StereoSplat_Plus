import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from PIL import Image
import cv2
import numpy as np
import PIL

def load_the_Metric3DV2_results(path,scale=256):
    # Read the image in unchanged mode (preserves uint16 format)
    img = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
    img = img/scale
    return img

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

import numpy as np

def img2cam_sparse(depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    将稀疏深度图转换为相机坐标系下的 3D 点。
    
    Args:
        depth_map: (H, W) 稀疏深度图（0 表示无效）
        K: (3, 3) 相机内参矩阵

    Returns:
        cam_points: (N, 3) 相机坐标系下的 3D 点（仅保留有效像素）
    """
    H, W = depth_map.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # 找出有效像素 (非零深度)
    valid_mask = depth_map > 0
    v, u = np.where(valid_mask)      # v: row index, u: col index
    z = depth_map[v, u]              # 有效深度值

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    cam_points = np.stack([x, y, z], axis=-1)  # (N, 3)
    return cam_points

def cam2image(points,K,height,width,depth_range=100):
    # camera points
    '''
    [3,N]
    '''
    ndim = points.ndim
    if ndim == 2:
        points = np.expand_dims(points, 0) #[1,3,N]

    points_proj = np.matmul(K[:3,:3].reshape([1,3,3]), points) #[1,3,N]
    depth = points_proj[:,2,:]
    depth[depth==0] = -1e-6
    u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.int32)
    v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.int32)

    if ndim==2:
        u = u[0]; v=v[0]; depth=depth[0]


    u = u.astype(np.int32)
    v = v.astype(np.int32)
    # prepare depth map for visualization
    depthMap = np.zeros((height, width))
    depthImage = np.zeros((height, width, 3))
    mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<width), v>=0), v<height)
    # visualize points within depth range meters
    mask = np.logical_and(np.logical_and(mask, depth>0), depth<depth_range)
    depthMap[v[mask],u[mask]] = depth[mask]
    
    return depthMap

def maybe_resize(src_reso, tgt_reso, ck):
    
    src_height, src_width = src_reso

    resize_flag = False
    if src_height != tgt_reso[0] or src_width != tgt_reso[1]:
        # img.resize((w, h))
        fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
        scale_h, scale_w = tgt_reso[0] / src_height, tgt_reso[1] / src_width
        fx_scaled, fy_scaled, cx_scaled, cy_scaled = fx * scale_w, fy * scale_h, cx * scale_w, cy * scale_h
        ck = np.array([[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]])
        resize_flag = True
    return ck, resize_flag

def resize_the_sparse_lidar(depthmap,raw_K,after_K,height,width,depth_range=100):
    
    points_cam = img2cam_sparse(depth_map=depthmap,K=raw_K)

    resize_gt_sparse_depth = cam2image(points=points_cam.T,K=after_K,height=height,width=width,
                                       depth_range=depth_range)
    
    return resize_gt_sparse_depth
    


if __name__=="__main__":
    
    # PseudoDepth_NMRFStereo
    reso = [224,840]
    gt_depth_path = "/data1/StereoDatasets/KITTI/KITTI360/projected_sparse_lidar/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000250.png"
    pseudo_depth_path = "/data1/StereoDatasets/KITTI/KITTI360/PseudoDepth_NMRFStereo/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000250.png"
    
    assert os.path.exists(pseudo_depth_path)
    gt_sparse_depth_data = load_the_Metric3DV2_results(gt_depth_path)
    gt_mask = gt_sparse_depth_data>0
    gt_mask = gt_mask.astype(np.float32)
    
    pseudo_depth_data = load_the_Metric3DV2_results(pseudo_depth_path)



    dptm = Image.fromarray(pseudo_depth_data)
    dptm = dptm.resize((reso[1], reso[0]), Image.BILINEAR)
    dptm = np.array(dptm)

    
    
    mse,mae = compute_depth_errors(gt_depth=gt_sparse_depth_data,
                                   aligned_depth=pseudo_depth_data,
                                   valid_mask=gt_mask.astype(np.bool_))
    
    '-----------------------------------------------------------------'
    
    raw_ck = np.array([[552.554261,   0,       682.049453],
                            [  0, 552.554261, 238.769549],
                            [  0, 0,    1]]) # 3x3 
    
    ck ,resize_flag = maybe_resize(src_reso=pseudo_depth_data.shape,tgt_reso=reso,ck=raw_ck)
    
    


    resize_gt_sparse_depth =resize_the_sparse_lidar(depthmap=gt_sparse_depth_data,
                                                    raw_K=raw_ck,
                                                    after_K=ck,
                                                    height=reso[0],
                                                    width=reso[1],
                                                    depth_range=100)
    
    
    mse_resize,mae_resize = compute_depth_errors(
        gt_depth=resize_gt_sparse_depth,
        aligned_depth=dptm,
        valid_mask=(resize_gt_sparse_depth>0)
    )


    
    pass



