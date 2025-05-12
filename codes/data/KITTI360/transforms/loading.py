import numpy as np
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
from model.utils.image import resize_image, HWC3
from mmdet3d.structures.points import LiDARPoints, BasePoints, get_points_type
from typing import List, Optional, Union
import cv2
import os.path as osp
import json
import copy
import matplotlib.pyplot as plt
import os



def load_info(info):
    img_path = info["data_path"]
    # use lidar coordinate of the key frame as the world coordinate
    c2w = info["sensor2lidar_transform"]
    
    
    # opencv cam -> opengl cam, maybe not necessary!
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w = c2w@flip_yz  # current c2w is the opengl coordinate

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



def load_conditions(img_paths, reso):
    

    

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
    depths = [] # depths normalized
    depths_m = [] # depths with metric
    confs_m = []  # confidence
    

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
            
        img = Image.open(img_path)
        h, w = img.height, img.width
        img, ck, resize_flag = maybe_resize(img, reso, raw_ck)
        
        img = HWC3(img)
        imgs.append(img)
        cks.append(ck)
        
        # relative depth from DepthAnything-v2
        # /data1/StereoDatasets/KITTI/KITTI360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000256.png
        # /data1/StereoDatasets/KITTI/KITTI360/monocular_depth/monodepthV2/data_2d_raw/2013_05_28_drive_0000_sync/
        depth_path = img_path.replace("data_2d_raw", "monocular_depth/monodepthV2/data_2d_raw")
        assert os.path.exists(depth_path)
        disp = load_the_depthanytingV2_results(depth_path)
   
        if resize_flag:
            disp = Image.fromarray(disp)
            disp = disp.resize((reso[1], reso[0]), Image.BILINEAR)
            disp = np.array(disp)

        
        # inverse disparity to relative depth
        # clamping the farthest depth to 50x of the nearest
        range = np.minimum(disp.max() / (disp.min() + 0.001), 50.0)
        max = disp.max()
        min = max / range
        depth = 1 / np.maximum(disp, min)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) # range from 0 ~1
        depths.append(depth)
        
        
        # metric depth from Metric3D-v2
        depthm_path = img_path.replace("data_2d_raw", "monocular_depth/Metric3DV2/data_2d_raw")
        depthm_path = depthm_path.replace(".png", "_dpt.png")
        conf_path = depthm_path.replace("_dpt.npy", "_conf.png")
        assert os.path.exists(depthm_path)
        assert os.path.exists(conf_path)
        
        dptm = load_the_Metric3DV2_results(depthm_path)
        conf = load_the_Metric3DV2_results(conf_path)
        
  
        if resize_flag:
            dptm = Image.fromarray(dptm)
            dptm = dptm.resize((reso[1], reso[0]), Image.BILINEAR)
            dptm = np.array(dptm)
            conf = Image.fromarray(conf)
            conf = conf.resize((reso[1], reso[0]), Image.BILINEAR)
            conf = np.array(conf)
        
        depths_m.append(dptm)
        confs_m.append(conf)

    imgs = torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float() / 255.0  # [v c h w]-->[6,3,H,W]
    depths = torch.from_numpy(np.stack(depths, axis=0)).float()  # [v h w] ---->[6,H,W]
    depths_m = torch.from_numpy(np.stack(depths_m, axis=0)).float()  # [v h w] ---->[6,H,W]
    confs_m = torch.from_numpy(np.stack(confs_m, axis=0)).float()  # [v h w]  ---->[6,H,W]
    cks = torch.as_tensor(cks, dtype=torch.float32)

    return imgs, depths, depths_m, confs_m, cks



def load_lidar_info(info):
    pcd_path = info["data_path"]
    lidar2sensor = np.eye(4)
    rot = info["sensor2lidar_rotation"]
    trans = info["sensor2lidar_translation"]
    lidar2sensor[:3, :3] = copy.deepcopy(rot.T)
    lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
    return pcd_path, lidar2sensor