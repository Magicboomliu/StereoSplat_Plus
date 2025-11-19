import torch
import torch.nn as nn
import torch.nn.functional as F
import mmengine
import pickle as pkl
import os
import sys
sys.path.append("..")
from projection.lidar import loadVelodyneData,Kitti360Viewer3DRaw
from projection.rotation import rotation_matrix_x,expand_to_4x4
from pcd_visualization import save_point_cloud_to_ply,visualize_point_cloud_with_axis
from pathlib import Path
import re
import open3d as o3d
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import cv2


import numpy as np
import cv2
from tqdm import tqdm


# def overlay_depth_on_rgb(rgb, depth, alpha=0.6, max_depth=80.0, colormap=cv2.COLORMAP_JET):
#     # 确保 RGB 是 uint8 格式
#     if rgb.dtype != np.uint8:
#         rgb = (rgb * 255).astype(np.uint8)

#     # 把 depth 归一化（忽略为 0 的无效点）
#     valid = depth > 0
#     depth_normalized = np.zeros_like(depth, dtype=np.uint8)
#     depth_clipped = np.clip(depth, 0, max_depth)
#     depth_normalized[valid] = (255 * depth_clipped[valid] / max_depth).astype(np.uint8)

#     # 应用 colormap
#     depth_colored = cv2.applyColorMap(depth_normalized, colormap)

#     # 融合 RGB 和 depth 伪彩色
#     overlay = rgb.copy()
#     overlay[valid] = cv2.addWeighted(rgb[valid], 1 - alpha, depth_colored[valid], alpha, 0)

#     return overlay



def overlay_depth_on_rgb_with_big_points(rgb, depth, radius=1, alpha=0.6, max_depth=80.0, colormap=cv2.COLORMAP_JET):
    rgb = rgb.copy()
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8)

    h, w = depth.shape
    valid = depth > 0
    ys, xs = np.where(valid)
    zs = np.clip(depth[ys, xs], 0, max_depth)
    depth_norm = (255 * zs / max_depth).astype(np.uint8)

    # 伪彩色
    depth_colors = cv2.applyColorMap(depth_norm, colormap)

    # 创建一个 overlay 层，画彩色大圆点
    overlay = rgb.copy()
    for (x, y, color) in zip(xs, ys, depth_colors):
        color = tuple(int(c) for c in np.array(color).flatten())
        cv2.circle(overlay, (x, y), radius, color, -1)

    # 将 overlay 与原图融合
    return cv2.addWeighted(rgb, 1 - alpha, overlay, alpha, 0)

def get_sequence_name(path):

    match = re.search(r'2013.*?sync', path)
    if match:
        result = match.group()
    
    return result

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



def cam2image(points,K):
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
    return u, v, depth

def get_projected_depth_map(points,K):

    u,v,depth = cam2image(points.T,K)
    u = u.astype(np.int32)
    v = v.astype(np.int32)
    height = 376
    width = 1408
    # prepare depth map for visualization
    depthMap = np.zeros((height, width))
    depthImage = np.zeros((height, width, 3))
    mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<width), v>=0), v<height)
    # visualize points within 30 meters
    mask = np.logical_and(np.logical_and(mask, depth>0), depth<30)
    depthMap[v[mask],u[mask]] = depth[mask]
    
    return depthMap
    
def read_image(path):
    
    image_data = np.array(Image.open(path).convert('RGB'))
     
    return image_data 
    


def visualize_multiple_lidar_pointclouds(lidar_list):
    vis_list = []

    # 使用 matplotlib colormap 分配颜色（例如 tab10 或 jet）
    cmap = plt.get_cmap('tab10')  # 有10种颜色，适合少量类别

    for idx, points in enumerate(lidar_list):
        points = points[:,:3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 映射颜色到 RGB (0~1)
        color = cmap(idx % 10)[:3]  # RGB tuple
        color_np = np.tile(np.array(color), (points.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(color_np)

        vis_list.append(pcd)

    o3d.visualization.draw_geometries(vis_list)


    
def visualize_pointcloud_and_cameras(point_cloud_xyz, cam_left_xyz, cam_right_xyz, radius=0.25):
    geometries = []

    # ✅ 1. 点云（随机颜色）
    pcd = o3d.geometry.PointCloud()
    point_cloud_xyz = np.asarray(point_cloud_xyz, dtype=np.float32).reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(point_cloud_xyz)
    colors = np.random.rand(point_cloud_xyz.shape[0], 3)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd)

    # ✅ 2. 左相机（红色球）
    for pos in cam_left_xyz:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        sphere.translate(pos)
        geometries.append(sphere)

    # ✅ 3. 右相机（蓝色球）
    for pos in cam_right_xyz:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color([0.0, 0.0, 1.0])
        sphere.translate(pos)
        geometries.append(sphere)

    # ✅ 4. 添加坐标轴（原点）
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geometries.append(coord)

    # ✅ 显示
    o3d.visualization.draw_geometries(geometries)
    

if __name__=="__main__":
    
    root_path = "/media/zliu/data12/dataset/KITTI/VSRD_Format/"
    
    bin_path = "/media/zliu/data12/dataset/KITTI/VSRD_Format/feedforward_bins/bin_infos/scene2013_05_28_drive_0000_sync_bin031.pkl"

    with open(bin_path, "rb") as f:
        bin_info = pkl.load(f)
        
    lidar_all = []
    cam_left_position_all = []
    cam_right_position_all = []
    
    for frame_idx in tqdm(range(len(bin_info['sensor_info']['LIDAR_TOP']))):
        

        center_frame_sensor_lidar = bin_info['sensor_info']['LIDAR_TOP'][frame_idx]
        center_frame_sensor_left = bin_info['sensor_info']['CAM_LEFT'][frame_idx]
        center_frame_sensor_right = bin_info['sensor_info']['CAM_RIGHT'][frame_idx]

        seq_name = get_sequence_frame_number(get_sequence_name(path=center_frame_sensor_lidar['data_path']))
        velo = Kitti360Viewer3DRaw(mode='velodyne', seq=seq_name,kitti360_path=root_path)
        instance_id = int(Path(os.path.basename(center_frame_sensor_lidar['data_path'])).stem)
        
        points = velo.loadVelodyneData(instance_id)
        points = velo.curlVelodyneData(instance_id, points)
        points[:,3] = 1 # lIDAR COORDINATE
        
        # to reference LiDAR
        points_in_reference = np.matmul(bin_info['sensor_info']['LIDAR_TOP'][frame_idx]['sensor2lidar_transform'],points.T).T
        lidar_all.append(points_in_reference)
        
        
        # To Left Cam 
        points_cam_left = np.matmul(np.linalg.inv(bin_info['sensor_info']["CAM_LEFT"][frame_idx]['sensor2lidar_transform']),
                                    points_in_reference.T).T
        points_cam_left = points_cam_left[:,:3]
        depthmap_left = get_projected_depth_map(points_cam_left,bin_info['sensor_info']["CAM_LEFT"][frame_idx]['camera_intrinsics'])
        left_image = read_image(os.path.join(root_path,bin_info['sensor_info']["CAM_LEFT"][frame_idx]['data_path']))
        
        cam_left_position = bin_info['sensor_info']["CAM_LEFT"][frame_idx]['sensor2lidar_transform'][:3,3]
        cam_left_position_all.append(cam_left_position)
        

        # TO Right CAM
        points_cam_right = np.matmul(np.linalg.inv(bin_info['sensor_info']["CAM_RIGHT"][frame_idx]['sensor2lidar_transform']),
                                    points_in_reference.T).T
        points_cam_right = points_cam_right[:,:3]
        depthmap_right = get_projected_depth_map(points_cam_right,bin_info['sensor_info']["CAM_RIGHT"][frame_idx]['camera_intrinsics'])
        right_image = read_image(os.path.join(root_path,bin_info['sensor_info']["CAM_RIGHT"][frame_idx]['data_path']))
        
        
        rgb_left_with_depth = overlay_depth_on_rgb_with_big_points(rgb=left_image,depth=depthmap_left)
        rgb_right_with_depth = overlay_depth_on_rgb_with_big_points(rgb=right_image,depth=depthmap_right)
        
        cam_right_position = bin_info['sensor_info']["CAM_RIGHT"][frame_idx]['sensor2lidar_transform'][:3,3]
        cam_right_position_all.append(cam_right_position)
        


    cam_left_position_all = [pos.reshape(1,3) for pos in cam_left_position_all]
    cam_right_position_all = [pos.reshape(1,3) for pos in cam_right_position_all]
    
    cam_left_position_all = np.vstack(cam_left_position_all)
    cam_right_position_all = np.vstack(cam_right_position_all)
    
    lidar_all = np.vstack(lidar_all)[:,:3]
    
    visualize_pointcloud_and_cameras(point_cloud_xyz=lidar_all,cam_left_xyz=cam_left_position_all,
                                     cam_right_xyz=cam_right_position_all)
    
    # visualize_point_cloud_with_axis(lidar_all,color=[0,0,1.0])
    # # visualize_point_cloud_with_axis(points[:,:3])

    

    

    



