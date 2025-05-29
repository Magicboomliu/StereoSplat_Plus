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

def read_annotation(annotation_filename):

    with open(annotation_filename) as file:
        annotation = json.load(file)

    extrinsic_matrix = torch.as_tensor(annotation["extrinsic_matrix"])
    
    return extrinsic_matrix

def load_condiations(annotation_path, reso,datapath, use_projected_lidar=True,use_pseudo_depth=True):
    imgs = []
    sparse_depths = []
    pseudo_depths = []
    cKs = [] # instrincs
    cTs = [] # extrincs

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
    
    
    current_filemname = annotation_path
    
    annotation_file_path_left = os.path.join(datapath,current_filemname)
    annotation_file_path_right = annotation_file_path_left.replace("image_00","image_01")
    assert os.path.exists(annotation_file_path_left), "Missing the Path is {}".format(annotation_file_path_left)
    assert os.path.exists(annotation_file_path_right),"Missing the Path is {}".format(annotation_file_path_right)

    left_image_path = current_filemname.replace("annotations",'data_2d_raw').replace(".json",".png")
    left_image_path_abs = os.path.join(datapath,left_image_path)
    
    right_image_path = left_image_path.replace("image_00","image_01")
    right_image_path_abs = os.path.join(datapath,right_image_path)
    assert os.path.exists(left_image_path_abs)
    assert os.path.exists(right_image_path_abs)
    
    # read the left and right images
    raw_ck = np.array([[552.554261,   0,       682.049453],
                            [  0, 552.554261, 238.769549],
                            [  0, 0,    1]]) # 3x3   
    left_image_data = Image.open(left_image_path_abs)
    right_image_data = Image.open(right_image_path_abs)
    h, w = left_image_data.height, right_image_data.width
    
    left_image_data, ck, resize_flag = maybe_resize(left_image_data, reso, raw_ck)
    right_image_data, ck, resize_flag = maybe_resize(right_image_data, reso, raw_ck)
    
    left_image_data = HWC3(left_image_data)
    imgs.append(left_image_data)
    cKs.append(ck) 
    
    right_image_data = HWC3(right_image_data)
    imgs.append(right_image_data)
    cKs.append(ck)
    
    # extrinics
    w2c_left = read_annotation(annotation_file_path_left)
    c2w_left = torch.inverse(w2c_left).cpu().numpy()
    w2c_right = read_annotation(annotation_file_path_right)
    c2w_right = torch.inverse(w2c_right).cpu().numpy()
    cTs.append(c2w_left)
    cTs.append(c2w_right)
    
    if use_projected_lidar:
        gt_projected_sparse_depth_path_left = current_filemname.replace("annotations","projected_sparse_lidar/data_2d_raw").replace(".json",".png")
        gt_projected_sparse_depth_path_right = gt_projected_sparse_depth_path_left.replace("image_00","image_01")
        
        gt_projected_sparse_depth_path_left_abs = os.path.join(datapath,gt_projected_sparse_depth_path_left)
        gt_projected_sparse_depth_path_right_abs = os.path.join(datapath,gt_projected_sparse_depth_path_right)
        
        assert os.path.exists(gt_projected_sparse_depth_path_left_abs)
        assert os.path.exists(gt_projected_sparse_depth_path_right_abs)
        
        sparse_gt_left = load_the_Metric3DV2_results(path=gt_projected_sparse_depth_path_left_abs) #(H,W)
        sparse_gt_right = load_the_Metric3DV2_results(path=gt_projected_sparse_depth_path_right_abs)


        if resize_flag:
            sparse_gt_left = Image.fromarray(sparse_gt_left)
            sparse_gt_left = sparse_gt_left.resize((reso[1], reso[0]), Image.BILINEAR)
            sparse_gt_left = np.array(sparse_gt_left)
            
            sparse_gt_right = Image.fromarray(sparse_gt_right)
            sparse_gt_right = sparse_gt_right.resize((reso[1], reso[0]), Image.BILINEAR)
            sparse_gt_right = np.array(sparse_gt_right)
        
        sparse_depths.append(sparse_gt_left)
        sparse_depths.append(sparse_gt_right)
        
    
    if use_pseudo_depth:
        # using nmrf-stereo by default
        psuedo_stereo_depth_left = current_filemname.replace("annotations","PseudoDepth_NMRFStereo/data_2d_raw").replace(".json",".png")
        psuedo_stereo_depth_right = psuedo_stereo_depth_left.replace("image_00","image_01")
        
        psuedo_stereo_depth_left_abs = os.path.join(datapath,psuedo_stereo_depth_left)
        psuedo_stereo_depth_right_abs = os.path.join(datapath,psuedo_stereo_depth_right)
        
        assert os.path.exists(psuedo_stereo_depth_left_abs)
        assert os.path.exists(psuedo_stereo_depth_right_abs)
        
        psuedo_depth_data_left = load_the_Metric3DV2_results(psuedo_stereo_depth_left_abs)
        psuedo_depth_data_right = load_the_Metric3DV2_results(psuedo_stereo_depth_right_abs)


        if resize_flag:
            psuedo_depth_data_left = Image.fromarray(psuedo_depth_data_left)
            psuedo_depth_data_left = psuedo_depth_data_left.resize((reso[1], reso[0]), Image.BILINEAR)
            psuedo_depth_data_left = np.array(psuedo_depth_data_left)
            
            psuedo_depth_data_right = Image.fromarray(psuedo_depth_data_right)
            psuedo_depth_data_right = psuedo_depth_data_right.resize((reso[1], reso[0]), Image.BILINEAR)
            psuedo_depth_data_right = np.array(psuedo_depth_data_right)
        
        pseudo_depths.append(psuedo_depth_data_left)
        pseudo_depths.append(psuedo_depth_data_right)
        
    imgs = torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float() / 255.0  # [v c h w]-->[6,3,H,W]
    cKs = torch.from_numpy(np.array(cKs)).float()
    cTs = torch.from_numpy(np.array(cTs)).float()
    
    depths_dict = dict()
    
    if use_projected_lidar:
        sparse_depths = torch.from_numpy(np.stack(sparse_depths, axis=0)).float()
        depths_dict['sparse_depths'] = sparse_depths
    
    if use_pseudo_depth:
        pseudo_depths = torch.from_numpy(np.stack(pseudo_depths, axis=0)).float() 
        depths_dict['pseudo_depths'] = pseudo_depths
    
    
    return imgs,cKs,cTs,depths_dict


def load_the_Metric3DV2_results(path,scale=256):
    # Read the image in unchanged mode (preserves uint16 format)
    img = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
    img = img/scale
    return img
