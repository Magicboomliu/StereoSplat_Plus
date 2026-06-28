import numpy as np
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
from stereosplat.model.utils.image import resize_image, HWC3
from mmdet3d.structures.points import LiDARPoints, BasePoints, get_points_type
from typing import List, Optional, Union
import cv2
import os.path as osp
import json
import copy
import matplotlib.pyplot as plt
import os


def load_info(info,cam_type='OpenGL'):
    img_path = info["data_path"]
    # use lidar coordinate of the key frame as the world coordinate
    c2w = info["sensor2lidar_transform"]
    
    
    if cam_type=='OpenGL':
        # opencv cam -> opengl cam, maybe not necessary!
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        c2w = c2w@flip_yz  # current c2w is the opengl coordinate
    else:
        c2w = c2w

    # lidar2cam rotatopnns
    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t
    
    return img_path, c2w, w2c

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

def resize_the_sparse_lidar(depthmap,raw_K,after_K,height,width,depth_range=100):
    
    points_cam = img2cam_sparse(depth_map=depthmap,K=raw_K)

    resize_gt_sparse_depth = cam2image(points=points_cam.T,K=after_K,height=height,width=width,
                                       depth_range=depth_range)
    
    return resize_gt_sparse_depth


def crop_size(h,w,img,K=None,type='pil'):
    
    # 光心位置
    cx, cy = 682.049453, 238.769549

    # 计算裁剪后尺寸（确保光心居中）
    crop_width = 2 * int(min(cx, w - cx))  # 1364
    crop_height = 2 * int(min(cy, h - cy))  # 274

    # 确定裁剪区域（优先裁右侧和上方）
    x_start = 0  # 左侧不裁
    x_end = x_start + crop_width  # 右侧裁到1364
    y_start = h - crop_height  # 从上方裁掉102行（376-274=102）
    y_end = h  # 保留下方所有行

    # 执行裁剪
    cropped_image = img[y_start:y_end, x_start:x_end]

    if type=='pil':
        cropped_image = Image.fromarray(cropped_image)
    
    # 更新内参
    new_cx = cx - x_start  # 682.049 - 0 = 682.049
    new_cy = cy - y_start  # 238.769 - 102 ≈ 136.769

    if K is not None:
        K[0,2] = new_cx
        K[1,2] = new_cy

        return K, cropped_image
    else:
        return cropped_image
    

def load_conditions(img_paths, reso,depth_info_params):
    
    def maybe_resize(img, tgt_reso, ck):
        if not isinstance(img, PIL.Image.Image):
            img = Image.fromarray(img)
        resize_flag = False
        if img.height != tgt_reso[0] or img.width != tgt_reso[1]:
            # img.resize((w, h))
            fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
            scale_h, scale_w = tgt_reso[0] / img.height, tgt_reso[1] / img.width
            fx_scaled, fy_scaled, cx_scaled, cy_scaled = fx * scale_w, fy * scale_h, cx * scale_w, cy * scale_h
            ck = np.array([[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]])
            img = img.resize((tgt_reso[1], tgt_reso[0]))
            resize_flag = True
        return np.array(img), ck, resize_flag
    
    
    imgs, cks = [], []
    
    if depth_info_params.use_pseudo_depth:
        depths = [] # depths normalized
        depths_m = [] # depths with metric
        confs_m = []  # confidence

    if depth_info_params.use_sparse_lidar:
        sparse_gt_depth_list = []

    for img_path in img_paths:      
        if "image_00/data_rect" in img_path:
            # left image
            raw_ck = np.array([[552.554261,   0,       682.049453],
                            [  0, 552.554261, 238.769549],
                            [  0, 0,    1]]) # 3x3            
        elif "image_01/data_rect" in img_path:
            # right image
            raw_ck = np.array([[552.554261,   0,       682.049453],
                            [  0, 552.554261, 238.769549],
                            [  0, 0,    1]]) # 3x3
        
        else:
            raise NotImplementedError
        
        # cropping to make sure the
        
        img = Image.open(img_path)
        h, w = img.height, img.width
        
        # crop the images for new cx,cy
        raw_ck, img  = crop_size(h=h,w=w,img=np.array(img),K=raw_ck)
        img, ck, resize_flag = maybe_resize(img, reso, raw_ck)

        
        img = HWC3(img)
        imgs.append(img)
        cks.append(ck)
        
        # relative depth from DepthAnything-v2
        # /data1/StereoDatasets/KITTI/KITTI360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000256.png
        # /data1/StereoDatasets/KITTI/KITTI360/monocular_depth/monodepthV2/data_2d_raw/2013_05_28_drive_0000_sync/

        if depth_info_params.use_pseudo_depth:
            depth_path = img_path.replace("data_2d_raw", "PseudoDepth_NMRFStereo/data_2d_raw")
            assert os.path.exists(depth_path)
            disp = load_the_Metric3DV2_results(depth_path)

            disp =  crop_size(h=disp.shape[0],w=disp.shape[1],img=disp,type='npy')
    
            
            if resize_flag:
                disp = Image.fromarray(disp)
                disp = disp.resize((reso[1], reso[0]), Image.BILINEAR)
                disp = np.array(disp)

            depth = disp
            depths.append(depth)


        if depth_info_params.use_sparse_lidar:
            # get sparse projected lidar
            sparse_gt_lidar_path = img_path.replace("data_2d_raw", "projected_sparse_lidar/data_2d_raw")
            assert os.path.exists(sparse_gt_lidar_path)
            sparse_gt_lidar_data = load_the_Metric3DV2_results(sparse_gt_lidar_path)
            
            # crop
            sparse_gt_lidar_data  = crop_size(h=sparse_gt_lidar_data.shape[0],
                                              w=sparse_gt_lidar_data.shape[1],
                                              img=sparse_gt_lidar_data,type='npy')
            
            if resize_flag:
                sparse_gt_lidar_data  = resize_the_sparse_lidar(depthmap=sparse_gt_lidar_data,
                                    raw_K=raw_ck,
                                    after_K=ck,
                                    height=reso[0],
                                    width=reso[1])
            
            sparse_gt_depth_list.append(sparse_gt_lidar_data)

        
        if depth_info_params.use_pseudo_depth:
            if depth_info_params.pseudo_depth_type=="Metric3DV2":
                # metric depth from Metric3D-v2
                depthm_path = img_path.replace("data_2d_raw", "monocular_depth/Metric3DV2/data_2d_raw")
                depthm_path = depthm_path.replace(".png", "_dpt.png")
                conf_path = depthm_path.replace("_dpt.npy", "_conf.png")
                assert os.path.exists(depthm_path)
                assert os.path.exists(conf_path)
                
                dptm = load_the_Metric3DV2_results(depthm_path)
                conf = load_the_Metric3DV2_results(conf_path)
                
                # crop
                dptm  = crop_size(h=dptm.shape[0],
                                              w=dptm.shape[1],
                                              img=dptm,type='npy')

                conf  = crop_size(h=conf.shape[0],
                                              w=conf.shape[1],
                                              img=conf,type='npy')       
                
                
                if resize_flag:
                    dptm = Image.fromarray(dptm)
                    dptm = dptm.resize((reso[1], reso[0]), Image.BILINEAR)
                    dptm = np.array(dptm)
                    conf = Image.fromarray(conf)
                    conf = conf.resize((reso[1], reso[0]), Image.BILINEAR)
                    conf = np.array(conf)
                
                depths_m.append(dptm)
                confs_m.append(conf)
                
            elif depth_info_params.pseudo_depth_type=="NMRFStereo":
                # metric depth from Metric3D-v2
                depthm_path = img_path.replace("data_2d_raw", "PseudoDepth_NMRFStereo/data_2d_raw")
                
                assert os.path.exists(depthm_path)
                dptm = load_the_Metric3DV2_results(depthm_path)
                conf = np.ones_like(dptm)

                # crop
                dptm  = crop_size(h=dptm.shape[0],
                                              w=dptm.shape[1],
                                              img=dptm,type='npy')

                conf  = crop_size(h=conf.shape[0],
                                              w=conf.shape[1],
                                              img=conf,type='npy') 
                

                
                if resize_flag:
                    dptm = Image.fromarray(dptm)
                    dptm = dptm.resize((reso[1], reso[0]), Image.BILINEAR)
                    dptm = np.array(dptm)
                    conf = Image.fromarray(conf)
                    conf = conf.resize((reso[1], reso[0]), Image.BILINEAR)
                    conf = np.array(conf)
                
                depths_m.append(dptm)
                confs_m.append(conf)
            
            else:
                raise NotImplementedError
        
        
        


    imgs = torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float() / 255.0  # [v c h w]-->[6,3,H,W]
    cks = torch.as_tensor(cks, dtype=torch.float32)
    
    
    if depth_info_params.use_pseudo_depth:
        depths = torch.from_numpy(np.stack(depths, axis=0)).float()  # [v h w] ---->[6,H,W]
        depths_m = torch.from_numpy(np.stack(depths_m, axis=0)).float()  # [v h w] ---->[6,H,W]
        confs_m = torch.from_numpy(np.stack(confs_m, axis=0)).float()  # [v h w]  ---->[6,H,W]
    else:
        depths = None
        depths_m = None
        confs_m = None
    
    if depth_info_params.use_sparse_lidar:
        sparse_gts = torch.from_numpy(np.stack(sparse_gt_depth_list,axis=0)).float()
    else:
        sparse_gts = None
    
    depth_dict = {"depths":depths,
                  "depths_m":depths_m,
                  "confs_m":confs_m,
                  "sparse_gts":sparse_gts
                  }
    

    
    return imgs, cks, depth_dict

def load_lidar_info(info):
    pcd_path = info["data_path"]
    lidar2sensor = np.eye(4)
    rot = info["sensor2lidar_rotation"]
    trans = info["sensor2lidar_translation"]
    lidar2sensor[:3, :3] = copy.deepcopy(rot.T)
    lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
    return pcd_path, lidar2sensor