import numpy as np
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
from .geometry_camera.image import resize_image,HWC3
from mmdet3d.structures.points import LiDARPoints, BasePoints, get_points_type
from typing import List, Optional, Union
import cv2
import os.path as osp
import json
import copy
import matplotlib.pyplot as plt
import os
import json


def Get_First_Key_Frame_LiDAR_To_World(datapath,simple_annotation_path):
    
    first_key_frame_simple_annotation_path = os.path.join(datapath,simple_annotation_path)
    
    data_info = load_json(first_key_frame_simple_annotation_path)
    
    left_cam_to_lidar_pose = np.array(data_info['left_cam_to_lidar'])
    left_cam_to_world_pose = np.array(data_info['left_cam_to_world'])

    lidar_to_world_pose  =left_cam_to_world_pose @ np.linalg.inv(left_cam_to_lidar_pose)
    return lidar_to_world_pose
    
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

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_image_data(img_path):
    img = Image.open(img_path)    
    return img

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

def preprocess_image_and_ck(image_data,raw_ck,reso):
    h, w = image_data.height, image_data.width
    raw_ck, image_data  = crop_size(h=h,w=w,img=np.array(image_data),K=raw_ck.copy())
    image_data, ck, resize_flag = maybe_resize(image_data, 
                                                    reso, 
                                                    raw_ck)
    image_data = HWC3(image_data)
    
    return image_data,raw_ck,ck,resize_flag

def preprocess_psuedo_depth(depth_data,reso,resize_flag):
    depth_data  = crop_size(h=depth_data.shape[0],
                            w=depth_data.shape[1],
                            img=depth_data,type='npy')
    
    if resize_flag:
        dptm = Image.fromarray(depth_data)
        dptm = dptm.resize((reso[1], reso[0]), Image.BILINEAR)
        dptm = np.array(dptm)
    
    return dptm
    
def preprocess_the_sparse_depth(sparse_gt_lidar_data,reso,
                                resize_flag,
                                raw_ck,ck):
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
    
    return sparse_gt_lidar_data

def get_timestep_infos(datapath,
                        annotation_path,
                       reso=[224,1088],
                       depth_info_params=None,
                       frist_ref=None,
                       extra_list=None):
    
    imgs, cks = [], []
    
    cam_to_lidar_pose_list = []
    lidar_to_world_pose_list = []
    
    data_dict = dict()
    data_info = load_json(annotation_path)
    
    left_image_data = load_image_data(os.path.join(datapath,data_info['left_image_path']))
    right_image_data = load_image_data(os.path.join(datapath,data_info['right_image_path']))
    raw_ck_input = np.array(data_info['raw_ck'])
    
    # left image preprocessing
    left_image_data,raw_ck,ck,resize_flag = preprocess_image_and_ck(image_data=left_image_data,
                            raw_ck=raw_ck_input,
                            reso=reso)
    imgs.append(left_image_data)
    cks.append(ck)

    # right image preprocessing
    right_image_data,raw_ck,ck,resize_flag = preprocess_image_and_ck(image_data=right_image_data,
                            raw_ck=raw_ck_input,
                            reso=reso)
    imgs.append(right_image_data)
    cks.append(ck)
    
    
    if depth_info_params.use_pseudo_depth:
        
        psuedo_depth_list = []
        left_image_psuedo_data = load_the_Metric3DV2_results(
            os.path.join(datapath,data_info['left_image_pseudo_depth']['stereo']))

        left_image_psuedo_data = preprocess_psuedo_depth(depth_data=left_image_psuedo_data,reso=reso,
                                resize_flag=resize_flag)
        
        psuedo_depth_list.append(left_image_psuedo_data)

        right_image_psuedo_data = load_the_Metric3DV2_results(
            os.path.join(datapath,data_info['right_image_pseudo_depth']['stereo']))

        right_image_psuedo_data = preprocess_psuedo_depth(depth_data=right_image_psuedo_data,reso=reso,
                                resize_flag=resize_flag)
        
        psuedo_depth_list.append(right_image_psuedo_data)
        

    if depth_info_params.use_sparse_lidar:
        
        gt_sparse_depth_list = []
    
        left_sparse_gt_lidar_data = load_the_Metric3DV2_results(
            os.path.join(datapath,data_info['left_image_pseudo_depth']['lidar']))
    
        left_sparse_gt_lidar_data = preprocess_the_sparse_depth(sparse_gt_lidar_data=left_sparse_gt_lidar_data,
                                                                resize_flag=resize_flag,
                                                                reso=reso,
                                                                raw_ck=raw_ck,
                                                                ck=ck)
        gt_sparse_depth_list.append(left_sparse_gt_lidar_data)
        
        right_sparse_gt_lidar_data = load_the_Metric3DV2_results(
            os.path.join(datapath,data_info['right_image_pseudo_depth']['lidar']))
    
        
        right_sparse_gt_lidar_data = preprocess_the_sparse_depth(sparse_gt_lidar_data=right_sparse_gt_lidar_data,
                                                                resize_flag=resize_flag,
                                                                reso=reso,
                                                                raw_ck=raw_ck,
                                                                ck=ck)
        
        gt_sparse_depth_list.append(right_sparse_gt_lidar_data)



    imgs = torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float() / 255.0  # [v c h w]-->[6,3,H,W]
    cks = torch.as_tensor(cks, dtype=torch.float32)
    
    input_cks = cks
    input_fxs, input_fys, input_cxs, input_cys = input_cks[:, 0, 0], input_cks[:, 1, 1], input_cks[:, 0, 2], input_cks[:, 1, 2]
    
    # compute image fovs and pixel directions
    input_fovxs, input_fovys = [], []
    # https://blog.csdn.net/OrdinaryMatthew/article/details/126670351
    for fx, fy, cx, cy in zip(input_fxs, input_fys, input_cxs, input_cys):
        fovx = 2 * np.arctan(cx / fx)
        fovy = 2 * np.arctan(cy / fy)
        input_fovxs.append(fovx)
        input_fovys.append(fovy)

    input_fovxs = torch.as_tensor(input_fovxs, dtype=torch.float32) #(6)
    input_fovys = torch.as_tensor(input_fovys, dtype=torch.float32) #(6)
        
    if depth_info_params.use_pseudo_depth:
        depths_ms = torch.from_numpy(np.stack(psuedo_depth_list, axis=0)).float()  # [v h w] ---->[6,H,W]
    else:
        depths_ms= None

    
    if depth_info_params.use_sparse_lidar:
        sparse_gts = torch.from_numpy(np.stack(gt_sparse_depth_list,axis=0)).float()
    else:
        sparse_gts = None
    
    if frist_ref is not None:
        world_to_ref_wolrd_pose  = np.linalg.inv(frist_ref) # world to ref LiDAR
    else:
        world_to_ref_wolrd_pose = np.eye(4,4)
    
    

    left_cam_to_lidar_pose = np.array(data_info['left_cam_to_lidar'])
    right_cam_to_lidar_pose = np.array(data_info['right_cam_to_lidar'])
    left_cam_to_world_pose = np.array(data_info['left_cam_to_world'])
    
    lidar_to_world_pose  =left_cam_to_world_pose @ np.linalg.inv(left_cam_to_lidar_pose)
    
    cam_to_lidar_pose_list.append(left_cam_to_lidar_pose)
    cam_to_lidar_pose_list.append(right_cam_to_lidar_pose)
    
    lidar_to_world_pose_list.append(lidar_to_world_pose)
    lidar_to_world_pose_list.append(lidar_to_world_pose)
    
    cam_to_lidar = torch.from_numpy(np.stack(cam_to_lidar_pose_list, axis=0)).float()
    lidar_to_cam = torch.linalg.inv(cam_to_lidar).float()
    lidar_to_cam = lidar_to_cam.transpose(1,2)
    

    input_w2is = []
    for w2c, ck in zip(lidar_to_cam, input_cks):
        viewpad = torch.eye(4)
        viewpad[:ck.shape[0], :ck.shape[1]] = ck
        w2i = (viewpad @ w2c.T)
        input_w2is.append(w2i)
    input_w2is = torch.stack(input_w2is) #(2,4,4), here is All Center
    

    # current LiDAR to true world
    lidar_to_world = torch.from_numpy(np.stack(lidar_to_world_pose_list,axis=0)).float()        
    world_to_ref_wolrd_pose = torch.from_numpy(world_to_ref_wolrd_pose).unsqueeze(0).repeat(lidar_to_world.shape[0],1,1)
    # shift to the Ref Frame LiDAR coordinate
    lidar_to_world = world_to_ref_wolrd_pose.float() @ lidar_to_world.float()
    # for rendering c2w
    output_cam2world = lidar_to_world @ cam_to_lidar

    data_dict['input'] = dict()
    data_dict['input']['imgs'] = imgs.float()
    data_dict['input']['cks'] = cks.float()
    data_dict['input']["fovxs"] = input_fovys.float()
    data_dict['input']['fovys'] = input_fovys.float()
    data_dict['input']['psuedo_depth'] = depths_ms.float()
    data_dict['input']["sparse_gts"] = sparse_gts.float()
    data_dict['input']['c2w'] = cam_to_lidar.float() # here in the ego cooordinate
    data_dict['input']['w2i'] = input_w2is.float()
    data_dict['input']['lidar_to_world'] = lidar_to_world.float() # for the post-processing for fusion


    
    data_dict['output'] = dict()
    data_dict['output']['psuedo_depth'] = depths_ms.float()
    data_dict['output']["sparse_gts"] = sparse_gts.float()
    data_dict['output']['imgs'] = imgs.float()
    data_dict['output']['cks'] = cks.float()
    data_dict['output']["fovxs"] = input_fovxs.float()
    data_dict['output']['fovys'] = input_fovys.float()
    data_dict['output']['c2w'] = output_cam2world.float()
    
    
    return data_dict

    
    
    

def get_inputs_info(datapath,
                    reso,
                    simple_annotation_path_list,
                    depth_info_params,
                    first_ref=None,
                    extra_list=None):
    '''extra list from '''
    

    return_list = []
    for annotation_file in simple_annotation_path_list:
        annotation_file_abs = os.path.join(datapath,annotation_file)
        data_dict = get_timestep_infos(
                        datapath=datapath,
                        annotation_path=annotation_file_abs,
                        reso = reso,
                        frist_ref=first_ref,
                        depth_info_params = depth_info_params,
                        extra_list=extra_list)

        return_list.append(data_dict)
    
    
    return return_list
    
    
    
    
    