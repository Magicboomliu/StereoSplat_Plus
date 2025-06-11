import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import sys
import argparse
import pickle as pkl
import open3d as o3d
from glob import glob
from tqdm import tqdm
from PIL import Image
import cv2

def load_the_Metric3DV2_results(path, scale=256):
    img = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
    img = img / scale
    return img

def compute_normals_from_depth(depth, fx, fy, cx, cy):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x3 = (x - cx) * z / fx
    y3 = (y - cy) * z / fy
    xyz = np.stack([x3, y3, z], axis=-1)
    dx = cv2.Sobel(xyz, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(xyz, cv2.CV_64F, 0, 1, ksize=3)
    normals = np.cross(dx, dy)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norms + 1e-8)
    return normals

def compute_rays(H, W, fx, fy, cx, cy):
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = (x - cx) / fx
    y = (y - cy) / fy
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays

def main():
    data_root_path = "/data1/StereoDatasets/KITTI/KITTI360/"
    example_bin_info = "/data1/StereoDatasets/KITTI/KITTI360/feedforward_bins/bin_infos_8.0/scene2013_05_28_drive_0000_sync_bin032.pkl"
    with open(example_bin_info, "rb") as f:
        bin_info = pkl.load(f)

    output_mesh_path = "tsdf_fusion_mesh.ply"
    output_pcd_path = "fusion_pcd.ply"

    sensor_info = bin_info['sensor_info']
    cam_left_info_list = sensor_info["CAM_LEFT"]
    left_cam_rgb_path_list = [os.path.join(data_root_path, d['data_path']) for d in cam_left_info_list]
    left_cam_depth_path_list = [p.replace("data_2d_raw", "PseudoDepth_NMRFStereo/data_2d_raw") for p in left_cam_rgb_path_list]
    left_cam_depth_data = [load_the_Metric3DV2_results(p) for p in left_cam_depth_path_list]
    left_extrinsics = [d['sensor2lidar_transform'] for d in cam_left_info_list]
    left_intrinsics = [d['camera_intrinsics'][:3, :3] for d in cam_left_info_list]

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i in tqdm(range(len(left_cam_depth_path_list)), desc="Integrating frames"):
        depth = left_cam_depth_data[i]
        color = o3d.io.read_image(left_cam_rgb_path_list[i])

        # -------- 裁剪上部1/4 ----------
        H_orig, W_orig = depth.shape
        crop_top = H_orig // 4
        depth = depth[crop_top:, :]
        color_np = np.array(color)[crop_top:, :, :]
        color = o3d.geometry.Image(color_np)
        H_new, W_new = depth.shape

        fx = left_intrinsics[i][0, 0]
        fy = left_intrinsics[i][1, 1]
        cx = left_intrinsics[i][0, 2]
        cy = left_intrinsics[i][1, 2] - crop_top  # 修正主点位置

        normals = compute_normals_from_depth(depth, fx, fy, cx, cy)
        rays = compute_rays(H_new, W_new, fx, fy, cx, cy)
        dot = np.abs(np.sum(normals * rays, axis=-1))
        angle = np.arccos(np.clip(dot, -1, 1))
        angle_deg = np.degrees(angle)
        mask = (angle_deg < 80) & (depth > 0)
        depth_filtered = depth.copy()
        depth_filtered[~mask] = 0

        depth_o3d = o3d.geometry.Image(depth_filtered.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_o3d,
            depth_scale=1.0,
            depth_trunc=80,
            convert_rgb_to_intensity=False)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=W_new, height=H_new,
            fx=fx, fy=fy, cx=cx, cy=cy)

        extrinsic = left_extrinsics[i]
        volume.integrate(rgbd, intrinsic, extrinsic)

    pcd = volume.extract_point_cloud()
    o3d.io.write_point_cloud(output_pcd_path, pcd)
    print(f"[✓] Point cloud saved to {output_pcd_path}")

if __name__ == "__main__":
    main()
