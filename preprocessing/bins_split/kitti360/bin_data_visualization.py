import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import pickle

from tqdm import tqdm

import numpy as np
import skimage.io
import pycocotools.mask
from utils.file_io import read_text_lines
from kitti360scripts.helpers.project import CameraPerspective
from projection.lidar import loadVelodyneData,Kitti360Viewer3DRaw
from projection.rotation import rotation_matrix_x,expand_to_4x4
import argparse
from PIL import Image
import re

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cv2
from tqdm import tqdm



def read_image(path):    
    return np.array(Image.open(path).convert("RGB")).astype(np.float32)
    
def load_pkl_file(path):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def save_results_into_uint16(results,scale_factor=256):
    results = results * scale_factor
    results = results.astype(np.uint16)    
    return results

def get_sequence_frame_number(sequence):
    if sequence=="2013_05_28_drive_0000_sync":
        return 0
    
    elif sequence =="2013_05_28_drive_0002_sync":
        return 2
    
    elif sequence=="2013_05_28_drive_0003_sync":
        return 3
    
    elif sequence == "2013_05_28_drive_0004_sync":
        return 4
    
    elif sequence == "2013_05_28_drive_0005_sync":
        return 5
    
    elif sequence == "2013_05_28_drive_0006_sync":
        return 6
    
    elif sequence == "2013_05_28_drive_0007_sync":
        return 7
    
    elif sequence == "2013_05_28_drive_0009_sync":
        return 9

    elif sequence == "2013_05_28_drive_0010_sync":
        return 10
    else:
        raise NotImplementedError
        
def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def get_seq_id_and_instance_id(path):    
    instance_id =  int(os.path.basename(path)[:-4])
    sequence_id = re.search(r"(2013_\d{2}_\d{2}_drive_\d{4}_sync)", path).group(1)
    return sequence_id,instance_id

def get_projected_depth(points,camera,TrVeloToRect,depth_range=80):
    
    # from LiDAR to Cam
    pointsCam = np.matmul(TrVeloToRect, points.T).T
    pointsCam = pointsCam[:,:3]
    
    
    u,v, depth= camera.cam2image(pointsCam.T)
    u = u.astype(np.int32)
    v = v.astype(np.int32)
    # prepare depth map for visualization
    depthMap = np.zeros((camera.height, camera.width))
    depthImage = np.zeros((camera.height, camera.width, 3))
    mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)
    # visualize points within depth range meters
    mask = np.logical_and(np.logical_and(mask, depth>0), depth<depth_range)
    depthMap[v[mask],u[mask]] = depth[mask]
    
    
    return depthMap

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


def draw_colored_sparse_depth_points_with_blending(image, depth, colormap='jet', point_size=3, alpha=1.0):
    """
    绘制彩色深度点，非深度区域做 alpha blending，整体更清晰
    
    Args:
        image: [H, W, 3] RGB 图像 (uint8)
        depth: [H, W] float32，0 表示无效
        colormap: matplotlib colormap 名（如 'turbo', 'jet'）
        point_size: int，点的半径（像素单位）
        alpha: float，深度点和原图的混合系数
    
    Returns:
        vis: [H, W, 3] 彩色图像（uint8）
    """
    assert image.shape[:2] == depth.shape, "image 和 depth 尺寸不一致"
    vis = image.copy().astype(np.float32)

    mask = depth > 0
    if not np.any(mask):
        return image

    # 归一化深度值
    d_min, d_max = depth[mask].min(), depth[mask].max()
    depth_norm = np.zeros_like(depth, dtype=np.float32)
    depth_norm[mask] = (depth[mask] - d_min) / (d_max - d_min + 1e-6)

    # 映射到伪彩色
    cmap = cm.get_cmap(colormap)
    depth_color = (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)

    # 创建彩色图层
    color_layer = np.zeros_like(vis, dtype=np.uint8)
    ys, xs = np.where(mask)
    for x, y in zip(xs, ys):
        color = tuple(int(c) for c in depth_color[y, x])
        cv2.circle(color_layer, (x, y), radius=point_size, color=color, thickness=-1, lineType=cv2.LINE_AA)

    # Alpha blending
    blended = vis.copy()
    color_layer = color_layer.astype(np.float32)
    mask_3ch = np.repeat(mask[:, :, None], 3, axis=2)
    blended[mask_3ch] = (
        alpha * color_layer[mask_3ch] + (1 - alpha) * vis[mask_3ch]
    )

    return blended.astype(np.uint8)


def concat_horizental(img1, img2):
    """
    垂直拼接两个图像：[H, W, 3] -> [2H, W, 3]
    
    Args:
        img1: 上半部分图像，shape=(H, W, 3)
        img2: 下半部分图像，shape=(H, W, 3)

    Returns:
        concat_img: 拼接后的图像，shape=(2H, W, 3)
    """
    assert img1.shape == img2.shape, "两张图像尺寸必须一致"
    return np.concatenate([img1, img2], axis=1)


def saved_json_files(dict_data,saved_path):
    with open(saved_path, "w") as f:
        json.dump(dict_data, f, indent=4)  # indent=4 让输出更可读

def main(args):
    
    os.makedirs(args.output_folder,exist_ok=True)
    
    if args.single_bin_token!="None":
        
    
        abs_bin_token_fname = args.single_bin_token
        bin_info = load_pkl_file(abs_bin_token_fname)
        
        token_name = bin_info['token'] # scene2013_05_28_drive_0000_sync_bin102
        saved_bin_folder = os.path.join(args.output_folder,token_name)
        os.makedirs(saved_bin_folder,exist_ok=True)
        scene_token_name = bin_info['scene_token'] # 2013_05_28_drive_0000_sync
        bin_length = bin_info['bin_length'] # 8.41656599159298
        
        saved_bin_info_dict = bin_info.copy()
        del saved_bin_info_dict['sensor_info']
        
        
        
        sensor_info_dict = bin_info['sensor_info'] # dict_keys(['LIDAR_TOP', 'CAM_LEFT', 'CAM_RIGHT'])
        
        
        left_images_data_list = []
        right_images_data_list = []
        
        left_images_with_projected_lidar_list = []
        right_images_with_projected_lidar_list = []
        
        sensor_info_left_images_dict = sensor_info_dict['CAM_LEFT']

        
        # left images and right image data
        for sensor_info in sensor_info_left_images_dict:
            data_path = sensor_info['data_path']
            
            saved_images_only_folder = os.path.join(saved_bin_folder,'images_only')
            saved_images_with_lidar = os.path.join(saved_bin_folder,'images_with_lidar')
            saved_depths = os.path.join(saved_bin_folder,"est_depths")
            saved_images_sep = os.path.join(saved_bin_folder,'images_sep')
            
            os.makedirs(saved_images_only_folder,exist_ok=True)
            os.makedirs(saved_images_with_lidar,exist_ok=True)
            os.makedirs(saved_depths,exist_ok=True)
            os.makedirs(saved_images_sep,exist_ok=True)
            
            
            if args.add_pseudo_depth:
                
                monodepth_folder_path = os.path.join(saved_depths,'monodepthv2')
                metric_depth_folder_path = os.path.join(saved_depths,'metricv2')
                os.makedirs(monodepth_folder_path,exist_ok=True)
                os.makedirs(metric_depth_folder_path,exist_ok=True)
            
            
            data_path = os.path.join(args.dataroot,data_path)
            
            left_image_data = read_image(data_path)
            left_images_data_list.append(left_image_data)
            
            right_image_data = read_image(data_path.replace("image_00","image_01"))
            right_images_data_list.append(right_image_data)
            
            sequence_id, instance_id = get_seq_id_and_instance_id(path=data_path)
        
            # calibration
            calibration_path = os.path.join(args.dataroot,"calibration/perspective.txt")
            assert os.path.exists(calibration_path)
            
            # velo lidar path
            velodyne_data_path = os.path.join(args.dataroot,'data_3d_raw',sequence_id,'velodyne_points','data',os.path.basename(data_path).replace(".png",".bin"))
            assert os.path.exists(velodyne_data_path)
            
            # left cam and right cam
            camera_left = CameraPerspective(args.dataroot, sequence_id, 0) # K,R, height, width: get all the intrincs
            camera_right = CameraPerspective(args.dataroot, sequence_id, 1) # K,R, height, width: get all the intrincs

            # velo
            seq_number = get_sequence_frame_number(sequence_id)
            velo_data = Kitti360Viewer3DRaw(mode='velodyne', seq=seq_number,kitti360_path=args.dataroot)

            # from velo to cam0, here we use using R_rect for rectification.
            TrVeloToRect0 = np.matmul(camera_left.R_rect, velo_data.TrVeloToCam['image_%02d' % 0]) #(4x4) 
            TrVeloToRect1 = np.matmul(camera_right.R_rect, velo_data.TrVeloToCam['image_%02d' % 1]) #(4x4)

            # raw point clouds
            # curl velodyne
            points = velo_data.loadVelodyneData(instance_id)    
            points = velo_data.curlVelodyneData(instance_id, points)
            points[:,3] = 1

            sparse_depth_map_left = get_projected_depth(points=points,camera=camera_left,
                                                 TrVeloToRect=TrVeloToRect0) # float64
            sparse_depth_map_right = get_projected_depth(points=points,camera=camera_right,
                                                         TrVeloToRect=TrVeloToRect1) # float64
            
            sparse_depth_map_left = sparse_depth_map_left
            left_with_project_lidar = draw_colored_sparse_depth_points_with_blending(left_image_data, sparse_depth_map_left)
            right_with_project_lidar = draw_colored_sparse_depth_points_with_blending(right_image_data,sparse_depth_map_right)
        
            left_images_with_projected_lidar_list.append(left_with_project_lidar)
            right_images_with_projected_lidar_list.append(right_with_project_lidar)

            left_right_image = concat_horizental(left_image_data,right_image_data).astype(np.uint8)
            left_right_with_projected_lidar_image = concat_horizental(left_with_project_lidar,right_with_project_lidar).astype(np.uint8)
                    
            skimage.io.imsave(os.path.join(saved_images_only_folder,os.path.basename(data_path)),left_right_image)
            skimage.io.imsave(os.path.join(saved_images_with_lidar,os.path.basename(data_path)),left_right_with_projected_lidar_image)
            
            skimage.io.imsave(os.path.join(saved_images_sep,os.path.basename(data_path).replace(".png","_l.png")),left_image_data.astype(np.uint8))
            skimage.io.imsave(os.path.join(saved_images_sep,os.path.basename(data_path).replace(".png","_r.png")),right_image_data.astype(np.uint8))

            
            saved_json_files(dict_data=saved_bin_info_dict,
                             saved_path=os.path.join(saved_bin_folder,"bin_info.json"))
            
            # get the monodepthV2
            
            if args.add_pseudo_depth:
                
                monodepth_path = data_path.replace("data_2d_raw", "monocular_depth/monodepthV2/data_2d_raw")
                assert os.path.exists(monodepth_path)
                disp = load_the_depthanytingV2_results(monodepth_path)
                range = np.minimum(disp.max() / (disp.min() + 0.001), 50.0)
                max = disp.max()
                min = max / range
                depth = 1 / np.maximum(disp, min)
                depth = (depth - depth.min()) / (depth.max() - depth.min()) # range from 0 ~1
                
                saved_monodepth_fname = os.path.join(monodepth_folder_path,os.path.basename(data_path))
                plt.imsave(saved_monodepth_fname,depth,cmap='jet')
                
                # load_the_depthanytingV2_results
                depthm_path = data_path.replace("data_2d_raw", "monocular_depth/Metric3DV2/data_2d_raw")
                depthm_path = depthm_path.replace(".png", "_dpt.png")
                conf_path = depthm_path.replace("_dpt.npy", "_conf.png")
                assert os.path.exists(depthm_path)
                assert os.path.exists(conf_path)
                
                dptm = load_the_Metric3DV2_results(depthm_path)
                conf = load_the_Metric3DV2_results(conf_path)
    
                dptm = dptm/(dptm.max()+1e-5)
                saved_metricdepth_fname = os.path.join(metric_depth_folder_path,os.path.basename(data_path))
                plt.imsave(saved_metricdepth_fname,dptm,cmap='jet')

            
    else:
        
        tokens_list = read_text_lines(args.bin_list)
        
        for token_name_rel in tqdm(tokens_list):
            abs_bin_token_fname = os.path.join(args.dataroot,"feedforward_bins",
                                               args.version,
                                               token_name_rel)
            
            bin_info = load_pkl_file(abs_bin_token_fname)

            token_name = bin_info['token'] # scene2013_05_28_drive_0000_sync_bin102
            saved_bin_folder = os.path.join(args.output_folder,token_name)
            os.makedirs(saved_bin_folder,exist_ok=True)
            scene_token_name = bin_info['scene_token'] # 2013_05_28_drive_0000_sync
            bin_length = bin_info['bin_length'] # 8.41656599159298
            
            saved_bin_info_dict = bin_info.copy()
            del saved_bin_info_dict['sensor_info']
            
            sensor_info_dict = bin_info['sensor_info'] # dict_keys(['LIDAR_TOP', 'CAM_LEFT', 'CAM_RIGHT'])
            
            left_images_data_list = []
            right_images_data_list = []
            
            left_images_with_projected_lidar_list = []
            right_images_with_projected_lidar_list = []
            
            sensor_info_left_images_dict = sensor_info_dict['CAM_LEFT']

            
            # left images and right image data
            for sensor_info in sensor_info_left_images_dict:
                data_path = sensor_info['data_path']
                
                saved_images_only_folder = os.path.join(saved_bin_folder,'images_only')
                saved_images_with_lidar = os.path.join(saved_bin_folder,'images_with_lidar')
                saved_depths = os.path.join(saved_bin_folder,"est_depths")
                saved_images_sep = os.path.join(saved_bin_folder,'images_sep')
                
                os.makedirs(saved_images_only_folder,exist_ok=True)
                os.makedirs(saved_images_with_lidar,exist_ok=True)
                os.makedirs(saved_depths,exist_ok=True)
                os.makedirs(saved_images_sep,exist_ok=True)
                
                
                if args.add_pseudo_depth:
                    
                    monodepth_folder_path = os.path.join(saved_depths,'monodepthv2')
                    metric_depth_folder_path = os.path.join(saved_depths,'metricv2')
                    os.makedirs(monodepth_folder_path,exist_ok=True)
                    os.makedirs(metric_depth_folder_path,exist_ok=True)
                
                
                data_path = os.path.join(args.dataroot,data_path)
                
                left_image_data = read_image(data_path)
                left_images_data_list.append(left_image_data)
                
                right_image_data = read_image(data_path.replace("image_00","image_01"))
                right_images_data_list.append(right_image_data)
                
                sequence_id, instance_id = get_seq_id_and_instance_id(path=data_path)
            
                # calibration
                calibration_path = os.path.join(args.dataroot,"calibration/perspective.txt")
                assert os.path.exists(calibration_path)
                
                # velo lidar path
                velodyne_data_path = os.path.join(args.dataroot,'data_3d_raw',sequence_id,'velodyne_points','data',os.path.basename(data_path).replace(".png",".bin"))
                assert os.path.exists(velodyne_data_path)
                
                # left cam and right cam
                camera_left = CameraPerspective(args.dataroot, sequence_id, 0) # K,R, height, width: get all the intrincs
                camera_right = CameraPerspective(args.dataroot, sequence_id, 1) # K,R, height, width: get all the intrincs

                # velo
                seq_number = get_sequence_frame_number(sequence_id)
                velo_data = Kitti360Viewer3DRaw(mode='velodyne', seq=seq_number,kitti360_path=args.dataroot)

                # from velo to cam0, here we use using R_rect for rectification.
                TrVeloToRect0 = np.matmul(camera_left.R_rect, velo_data.TrVeloToCam['image_%02d' % 0]) #(4x4) 
                TrVeloToRect1 = np.matmul(camera_right.R_rect, velo_data.TrVeloToCam['image_%02d' % 1]) #(4x4)

                # raw point clouds
                # curl velodyne
                points = velo_data.loadVelodyneData(instance_id)    
                points = velo_data.curlVelodyneData(instance_id, points)
                points[:,3] = 1

                sparse_depth_map_left = get_projected_depth(points=points,camera=camera_left,
                                                    TrVeloToRect=TrVeloToRect0) # float64
                sparse_depth_map_right = get_projected_depth(points=points,camera=camera_right,
                                                            TrVeloToRect=TrVeloToRect1) # float64
                
                sparse_depth_map_left = sparse_depth_map_left
                left_with_project_lidar = draw_colored_sparse_depth_points_with_blending(left_image_data, sparse_depth_map_left)
                right_with_project_lidar = draw_colored_sparse_depth_points_with_blending(right_image_data,sparse_depth_map_right)
            
                left_images_with_projected_lidar_list.append(left_with_project_lidar)
                right_images_with_projected_lidar_list.append(right_with_project_lidar)

                left_right_image = concat_horizental(left_image_data,right_image_data).astype(np.uint8)
                left_right_with_projected_lidar_image = concat_horizental(left_with_project_lidar,right_with_project_lidar).astype(np.uint8)
                        
                skimage.io.imsave(os.path.join(saved_images_only_folder,os.path.basename(data_path)),left_right_image)
                skimage.io.imsave(os.path.join(saved_images_with_lidar,os.path.basename(data_path)),left_right_with_projected_lidar_image)
                
                skimage.io.imsave(os.path.join(saved_images_sep,os.path.basename(data_path).replace(".png","_l.png")),left_image_data.astype(np.uint8))
                skimage.io.imsave(os.path.join(saved_images_sep,os.path.basename(data_path).replace(".png","_r.png")),right_image_data.astype(np.uint8))

                
                saved_json_files(dict_data=saved_bin_info_dict,
                                saved_path=os.path.join(saved_bin_folder,"bin_info.json"))
                
                # get the monodepthV2
                
                if args.add_pseudo_depth:
                    
                    monodepth_path = data_path.replace("data_2d_raw", "monocular_depth/monodepthV2/data_2d_raw")
                    assert os.path.exists(monodepth_path)
                    disp = load_the_depthanytingV2_results(monodepth_path)
                    range = np.minimum(disp.max() / (disp.min() + 0.001), 50.0)
                    max = disp.max()
                    min = max / range
                    depth = 1 / np.maximum(disp, min)
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) # range from 0 ~1
                    
                    saved_monodepth_fname = os.path.join(monodepth_folder_path,os.path.basename(data_path))
                    plt.imsave(saved_monodepth_fname,depth,cmap='jet')
                    
                    # load_the_depthanytingV2_results
                    depthm_path = data_path.replace("data_2d_raw", "monocular_depth/Metric3DV2/data_2d_raw")
                    depthm_path = depthm_path.replace(".png", "_dpt.png")
                    conf_path = depthm_path.replace("_dpt.npy", "_conf.png")
                    assert os.path.exists(depthm_path)
                    assert os.path.exists(conf_path)
                    
                    dptm = load_the_Metric3DV2_results(depthm_path)
                    conf = load_the_Metric3DV2_results(conf_path)
        
                    dptm = dptm/(dptm.max()+1e-5)
                    saved_metricdepth_fname = os.path.join(metric_depth_folder_path,os.path.basename(data_path))
                    plt.imsave(saved_metricdepth_fname,dptm,cmap='jet')
            

    
    

if __name__=="__main__":
    
    # Training settings
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--version', type=str)
    parser.add_argument('--bin_list', type=str)
    parser.add_argument('--single_bin_token', type=str)
    parser.add_argument('--output_folder', type=str)
    
    
    parser.add_argument('--add_pseudo_depth',  action="store_true")
    
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus

    main(args)