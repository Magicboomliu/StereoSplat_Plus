import os
import time
import json
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union
import math
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import numpy as np
import skimage
import pycocotools.mask
from utils.file_io import read_text_lines
from kitti360scripts.helpers.project import CameraPerspective
from projection.lidar import loadVelodyneData,Kitti360Viewer3DRaw
from projection.rotation import rotation_matrix_x,expand_to_4x4

from tqdm import tqdm
import mmengine

import open3d as o3d

def depth_to_pointcloud_torch(depth_map, K, extrinsic):
    """
    depth_map: torch.Tensor, shape [H, W], values in meters
    K: torch.Tensor, shape [3, 3] (intrinsic)
    extrinsic: torch.Tensor, shape [4, 4] (cam-to-world transform)
    return: [N, 3] point cloud in world coords
    """
    device = depth_map.device
    H, W = depth_map.shape

    # Create meshgrid of image coordinates
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = x.float()
    y = y.float()

    # Only valid (non-zero) depth
    valid_mask = depth_map > 0
    z = depth_map[valid_mask]
    x = x[valid_mask]
    y = y[valid_mask]

    # Project to camera space
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy

    ones = torch.ones_like(z)
    points_cam = torch.stack([x_cam, y_cam, z, ones], dim=1)  # [N, 4]
    

    # 应用C2W变换矩阵
    world_points_hom = points_cam.cpu().numpy() @ extrinsic.cpu().numpy().T  # (N, 4)
    # 转换回三维坐标
    world_points = world_points_hom[:, :3] / world_points_hom[:, 3, np.newaxis]

    points_world = torch.from_numpy(world_points)
    return points_world.cpu()

import open3d as o3d
import torch
import numpy as np
import re
from pyquaternion import Quaternion
from pathlib import Path

def visualize_pointcloud_with_axes(points_world, axis_length=10.0, line_width=5.0):
    """
    points_world: torch.Tensor of shape [N, 3]
    axis_length: length of XYZ axis lines
    line_width: thickness of axis lines
    """
    # Convert to numpy
    points_np = points_world.cpu().numpy()

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Compute center
    center = points_np.mean(axis=0)

    # XYZ axis lines
    axis_points = np.array([
        center, center + [axis_length, 0, 0],   # X
        center, center + [0, axis_length, 0],   # Y
        center, center + [0, 0, axis_length],   # Z
    ])
    lines = [[0, 1], [2, 3], [4, 5]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axis_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Open3D visualizer with line width
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    render_option = vis.get_render_option()
    render_option.line_width = line_width  # <-- 设置线宽
    vis.run()
    vis.destroy_window()


camera_types = [
    "CAM_LEFT",
    "CAM_RIGHT",]

def get_c2ws(cam0_to_pose_path):
    camera_extrinsics_list = np.loadtxt(cam0_to_pose_path, dtype=np.float32)#(N,17)
    camera_extrinsics_dict = dict()
    for sample in camera_extrinsics_list:
        camera_extrinsics_dict[int(sample[0])] = sample[1:].reshape(4,4).astype(np.float32)
    return camera_extrinsics_dict

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

def loaded_sensors_path_info(root_path,annotation_path):
    
    sequence_name,cam_id_left,instance_id = get_sequence_name_and_cam_id(annotation_path)
    
    # image_path
    left_image_name = os.path.join(root_path, "data_2d_raw/{}/image_00/data_rect/{:010d}.png".format(sequence_name, int(instance_id)))
    right_image_name = os.path.join(root_path, "data_2d_raw/{}/image_01/data_rect/{:010d}.png".format(sequence_name, int(instance_id)))
    
    assert os.path.exists(left_image_name)
    assert os.path.exists(right_image_name)
    
    # annotation_path(COCO Format)
    left_annotation_path = annotation_path
    right_annotation_path = left_annotation_path.replace("image_00","image_01")
    

    
    assert os.path.exists(left_annotation_path)
    assert os.path.exists(right_annotation_path)
    
    # calibration path: K cam
    calibration_path = os.path.join(root_path, "calibration/perspective.txt")
    assert os.path.exists(calibration_path)
    
    # camera pose.
    cam0_pose_path = os.path.join(root_path,"data_poses/{}/cam0_to_world.txt".format(sequence_name))
    assert os.path.exists(cam0_pose_path)
    
    # Velodyne Path: LiDAR
    velodyne_path = os.path.join(root_path,"data_3d_raw/{}/velodyne_points/data/{:010d}.bin".format(sequence_name,int(instance_id)))
    assert os.path.exists(velodyne_path)
    
    
    return left_image_name,right_image_name,left_annotation_path,right_annotation_path,calibration_path,cam0_pose_path,velodyne_path,sequence_name,instance_id

def loaded_sensors_data_info(root_path,annotation_path):
    
    left_image_name,right_image_path,left_annotation_path,right_annotation_path,calibration_path,cam0_pose_path,velodyne_path,sequence_name,instance_id = \
                                loaded_sensors_path_info(root_path=root_path,annotation_path=annotation_path)
    
    # loaded current camear info
    left_camera = CameraPerspective(root_path, sequence_name,0) # K,R, height, width: get all the intrincs    
    right_camera = CameraPerspective(root_path,sequence_name,1)
    
    # left instrinics
    left_cam_intrinsic = left_camera.K        # 3x4
    
    # right instrincs
    right_cam_instrinsic = right_camera.K     # -baseline × f_x
    
    
    # left cam to world
    left_cam2world = left_camera.cam2world[instance_id]     # left cam to world
    # right cam to world
    right_cam2world = right_camera.cam2world[instance_id]   # left cam to world
    
    sequence_number_id = get_sequence_frame_number(sequence_name)
    velo = Kitti360Viewer3DRaw(mode = 'velodyne', seq = sequence_number_id, kitti360_path=args.root_path)
    
    # velo to cam0
    TrVeloToRectCam0 = np.matmul(left_camera.R_rect, velo.TrVeloToCam['image_%02d' %0]) #(4x4)
    # cam0 to velo
    RectCam0ToVelo = np.linalg.inv(TrVeloToRectCam0)

    # velo to cam1
    TrVeloToRectCam1 = np.matmul(right_camera.R_rect, velo.TrVeloToCam['image_%02d' %1]) #(4x4)
    # cam0 to velo
    RectCam1ToVelo = np.linalg.inv(TrVeloToRectCam1)
    
    
    return_dict = {
        "left_cam_intrinsic":left_cam_intrinsic, # Left Instrinsic
        "left_cam_to_world": left_cam2world,    # Left Cam To World
        "left_cam_to_velo": RectCam0ToVelo,     # Left To Velo
        "velo_to_left_cam": TrVeloToRectCam0,   # Velo To Left
        "right_cam_intrinsic":right_cam_instrinsic,  # Right Instrinsic
        "right_cam_to_world":right_cam2world,        # Right Cam To World
        "right_cam_to_velo": RectCam1ToVelo,        # Right Cam To Velo
        "velo_to_right_cam": TrVeloToRectCam1       # Velo To Right
    }
    
    
    return return_dict

def rotation_matrix_x(angles):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    one = torch.ones_like(angles)
    zero = torch.zeros_like(angles)
    rotation_matrices = torch.stack([
        torch.stack([ one, zero,  zero], dim=-1),
        torch.stack([zero,  cos,  -sin], dim=-1),
        torch.stack([zero,  sin,   cos], dim=-1),
    ], dim=-2)
    return rotation_matrices

def expand_to_4x4(matrices):
    matrices_4x4 = torch.eye(4).to(matrices)
    matrices_4x4 = matrices_4x4.reshape(*[1] * len(matrices.shape[:-2]), 4, 4)
    matrices_4x4 = matrices_4x4.repeat(*matrices.shape[:-2], 1, 1)
    matrices_4x4[..., :matrices.shape[-2], :matrices.shape[-1]] = matrices
    return matrices_4x4

def read_annotation(annotation_filename,class_names=['car']):

    with open(annotation_filename) as file:
        annotation = json.load(file)

    intrinsic_matrix = torch.as_tensor(annotation["intrinsic_matrix"])
    extrinsic_matrix = torch.as_tensor(annotation["extrinsic_matrix"])

    instance_ids = {
        class_name: list(masks.keys())
        for class_name, masks in annotation["masks"].items()
        if class_name in class_names
    }

    if instance_ids:

        masks = torch.cat([
            torch.as_tensor(np.stack([
                pycocotools.mask.decode(annotation["masks"][class_name][instance_id])
                for instance_id in instance_ids
            ]), dtype=torch.float)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        labels = torch.cat([
            torch.as_tensor([class_names.index(class_name)] * len(instance_ids), dtype=torch.long)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        boxes_3d = torch.cat([
            torch.as_tensor([
                annotation["boxes_3d"][class_name].get(instance_id, [[np.nan] * 3] * 8)
                for instance_id in instance_ids
            ], dtype=torch.float)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        instance_ids = torch.cat([
            torch.as_tensor(list(map(int, instance_ids)), dtype=torch.long)
            for instance_ids in instance_ids.values()
        ], dim=0)

        return dict(
            masks=masks,
            labels=labels,
            boxes_3d=boxes_3d,
            instance_ids=instance_ids,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
        )

    else:

        return dict(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
        )

def get_sequence_name_and_cam_id(annotation_path):
    
    # current sequence name
    sequence_name = re.search(r"(2013_\d{2}_\d{2}_drive_\d{4}_sync)", annotation_path).group(1)
    instance_id = os.path.basename(annotation_path)[:-5]
    cam_id = re.search(r"/(image_\d{2})/", annotation_path).group(1)
    
    if cam_id =="image_00":
        cam_id = 0
    else:
        cam_id = 1
    
    return sequence_name,cam_id,int(instance_id)


def obtain_sensor2_reference_lidar_top(args,
                                       current_annotation_path,
                                       reference_dict,
                                       sensor_type):
            
    sensors_info_dict = loaded_sensors_data_info(root_path=args.root_path,annotation_path=current_annotation_path)    
    

    
    if sensor_type=="LIDAR_TOP":   
        data_path = current_annotation_path.replace("annotations","data_3d_raw").replace("image_00/data_rect","velodyne_points/data").replace(".json",".bin")
    elif sensor_type=="CAM_LEFT":
        data_path = current_annotation_path.replace("annotations","data_2d_raw").replace(".json",'.png')
    elif sensor_type=="CAM_RIGHT":
        data_path = current_annotation_path.replace("annotations","data_2d_raw").replace(".json",'.png').replace("image_00","image_01")
    else:
        raise NotImplementedError
    
    assert os.path.exists(data_path), "data file {} not exist!".format(data_path)
    data_path = data_path[len(args.root_path):]

    # to KITTI-360 World Coodrinate
    if sensor_type=="LIDAR_TOP":
        sensor_to_world = np.matmul(sensors_info_dict['left_cam_to_world'],sensors_info_dict['velo_to_left_cam'])
    elif sensor_type=="CAM_LEFT":
        sensor_to_world = sensors_info_dict['left_cam_to_world']
    elif sensor_type=="CAM_RIGHT":
        sensor_to_world = sensors_info_dict['right_cam_to_world']
    
    sensor_to_world_rotation = sensor_to_world[:3,:3]
    sensor_to_world_translation = sensor_to_world[:3,3]
    sensor_to_world_transform = np.eye(4)
    sensor_to_world_transform[:3,:3] = sensor_to_world_rotation 
    sensor_to_world_transform[:3,3] = sensor_to_world_translation
    
    # To Reference View LiDAR Coordinate
    sensor_to_reference_cam0 = np.linalg.inv(reference_dict['left_cam_to_world']) @ sensor_to_world
    sensor_to_refernece_lidar = np.matmul(reference_dict['left_cam_to_velo'],sensor_to_reference_cam0)
    sensor_to_refernece_lidar_rotation = sensor_to_refernece_lidar[:3,:3]
    sensor_to_reference_lidar_translation = sensor_to_refernece_lidar[:3,3]
    sensor_to_reference_lidar_transform = np.eye(4)
    sensor_to_reference_lidar_transform[:3,:3] = sensor_to_refernece_lidar_rotation
    sensor_to_reference_lidar_transform[:3,3] = sensor_to_reference_lidar_translation


    sensors_processed_info = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_token":Path(os.path.basename(data_path)).stem,
        "sensor2world_translation":sensor_to_world_translation,
        "sensor2world_rotation": sensor_to_world_rotation,
        "sensor2world_transform": sensor_to_world_transform,
        "sensor2lidar_translation":sensor_to_reference_lidar_translation,
        "sensor2lidar_rotation":sensor_to_refernece_lidar_rotation,
        "sensor2lidar_transform":sensor_to_reference_lidar_transform,
        "sensor2_reference_cam0": sensor_to_reference_cam0,
        
    }
    
    if "LEFT" in sensor_type:
        sensors_processed_info['camera_intrinsics']= sensors_info_dict['left_cam_intrinsic']
    elif "RIGHT" in sensor_type:
        sensors_processed_info['camera_intrinsics']= sensors_info_dict['right_cam_intrinsic']
    
    
    return sensors_processed_info
        

def generate_bin_info(args,bin_token,scene_token,samples_all,bin_start, bin_end, bin_center, bin_length):
    assert bin_end > bin_start
    
    first_sample_annotation_path = samples_all[bin_start]
    last_sample_annotation_path = samples_all[bin_end]
    center_sample_annotation_path = samples_all[bin_center]
    
    
    # using the first view as the reference view
    reference_sample_infos_dict = loaded_sensors_data_info(root_path=args.root_path,
                                                      annotation_path=first_sample_annotation_path)
    

    # bin samples is order def
    bin_samples_annotation_path = [center_sample_annotation_path, first_sample_annotation_path, last_sample_annotation_path] + \
        samples_all[bin_start+1:bin_center] + \
        samples_all[bin_center+1:bin_end]
        
    info = {    
        "token": bin_token,
        "scene_token": scene_token,
        "timestep":os.path.basename(center_sample_annotation_path)[:-4],
        "bin_length": bin_length,
        "original_reference_view_info_dict": reference_sample_infos_dict
    }
    
    # {'LIDAR_TOP': [], 'CAM_LEFT': [], 'CAM_RIGHT': []}
    sensor_info = {k:[] for k in ["LIDAR_TOP"] + camera_types}
    
    for sample_path in bin_samples_annotation_path:    
        for sensor in sensor_info.keys():
            current_sensor_info = obtain_sensor2_reference_lidar_top(
                                            args=args,
                                            current_annotation_path=sample_path,
                                            reference_dict=reference_sample_infos_dict,
                                            sensor_type=sensor)
            sensor_info[sensor].append(current_sensor_info)
    
    info.update(sensor_info=sensor_info)
    
    return info
            
      
def create_kitti_infos(args,annotation_path,current_seq_name):
    
    def find_bin_end(bin_start,dists,min_bin_length):
        '''
        dists_cp: 复制 dists，用于遍历。
        dist_acc: 距离累计值。

        bin_end: 当前累计到第几个帧。

        bin_center: 记录 bin 中点（当累积距离 >= 一半时设定）。

        flag: 最终是否找到了合法 bin。

        center_flag: 防止重复设置中点。
        
        '''

        
        dists_cp = list(dists)
        dist_acc = 0
        bin_end = 0
        bin_center = 0
        flag = False
        center_flag = False
        while len(dists_cp) > 0:
            dist_acc += dists_cp.pop(0)
            bin_end += 1
            if dist_acc >= min_bin_length / 2 and not center_flag:
                bin_center = bin_end
                center_flag = True
            if dist_acc >= min_bin_length:
                flag = True
                break
        # 找到了对应的长度
        if flag:
            return bin_end + bin_start, bin_center + bin_start
        else:
            return None, None
             
    assert os.path.exists(args.out_dir)
    assert args.out_dir is not None
    
    
    all_the_bins = {"bins": [], "adjacent_bins": []}
    train_scene_tokens = []

    annotations_list = read_text_lines(annotation_path)# 10238
    annotations_list = [os.path.join(args.root_path,f) for f in annotations_list]
    
    # DEBUG: FIXME
    annotations_list = annotations_list
    
    min_bin_length = args.min_bin_length

    dists = []
    
    for i in tqdm(range(len(annotations_list) - 1)):
        
        sample_0 = annotations_list[i]
        sample_1 = annotations_list[i + 1]
        
        # Left Cam to World at Sample 0
        left_cam2world_0 = loaded_sensors_data_info(args.root_path, sample_0)['left_cam_to_world']
        # Left Cam to World at Smaple 1
        left_cam2world_1 = loaded_sensors_data_info(args.root_path, sample_1)['left_cam_to_world']

        translation_sample_0 = left_cam2world_0[:3,3]
        sample_x0,sample_y0 = translation_sample_0[:2]
        
        translation_sample_1 = left_cam2world_1[:3,3]
        sample_x1,sample_y1 = translation_sample_1[:2]
        # car move in the x-y direction

        dist_i = math.sqrt((sample_x0 - sample_x1) ** 2 + (sample_y0 - sample_y1) ** 2)
        dists.append(dist_i)
    
    dist_sum = sum(dists) # get th summs
    assert len(dists) == len(annotations_list) -1
    
    # split bins here
    bin_id = 0
    
    # if current sequences is bigger than min_bin_length
    if dist_sum>=min_bin_length:
        bin_start, bin_end = 0, 0
        bin_tokens = [] # save all in this scences
        
        current_finished_files = 0
        while bin_start < len(annotations_list):
            # find one bin
            bin_end, bin_center = find_bin_end(bin_start,list(dists[bin_start:]), min_bin_length)
            
            if bin_end is not None:
                bin_token = "scene{}_bin{:03d}".format(current_seq_name, bin_id) # give a token names
                
                bin_info = generate_bin_info(args,bin_token,current_seq_name,annotations_list,bin_start,bin_end,bin_center,sum(dists[bin_start:bin_end]))
                bin_start += 1
                bin_id += 1
                bin_tokens.append(bin_token)

                bin_filename = osp.join(args.out_dir, "bin_infos_{}".format(str(args.min_bin_length)), "{}.pkl".format(bin_token))
                mmengine.dump(bin_info, bin_filename)
                all_the_bins['bins'].append(bin_token)
            
            else:
                break
        
        current_finished_files = current_finished_files +1
        print("current Finished saved {}".format(current_finished_files))
            
        
        all_the_bins["adjacent_bins"]= bin_tokens

    
        
    else:
        _, bin_center = find_bin_end(bin_start,list(dists[bin_start:]), min_bin_length)
        bin_token = "scene{}_bin{:03d}".format(current_seq_name, bin_id) # give a token names
        bin_info = generate_bin_info(args,bin_token,current_seq_name,annotations_list,
                                     0,len(annotations_list)-1,bin_center,dist_sum)
        bin_id += 1
        bin_filename = osp.join(args.out_dir, "bin_infos_{}".format(str(args.min_bin_length)), "{}.pkl".format(bin_token))
        mmengine.dump(bin_info, bin_filename)

        all_the_bins["bins"].append(bin_token)
        all_the_bins["adjacent_bins"].append([bin_token])
 
        
def kitti360_data_prep(args):
    idx = 0
    all_sequence_names = sorted(os.listdir(args.filelist_folder))
    all_sequence_names = all_sequence_names[:1]
    
    for filename_list in all_sequence_names:
        seq_name = os.path.basename(filename_list)[:-9]
        print("Processed Current Seq is {}, Finished {}/{}".format(seq_name,idx,len(os.listdir(args.filelist_folder))))
        idx = idx +1    
        current_annotations_fname = os.path.join(args.filelist_folder, filename_list)
        create_kitti_infos(args=args,annotation_path=current_annotations_fname,current_seq_name=seq_name)        

    print("All Has been Finished")


if __name__=="__main__":
        
    import argparse
    parser = argparse.ArgumentParser(description="Data converter arg parser")

    parser.add_argument(
        "--root_path",
        type=str,
        required=False,
        default="/media/zliu/data12/dataset/KITTI/VSRD_Format/",
        help="specify the root path of dataset",
    )
    
    parser.add_argument(
        "--filelist_folder",
        type=str,
        required=False,
        default="/home/zliu/Desktop/Project2025/KITTI360_for_feedforward/Preprocessing/filelist",
        help="specify the root path of dataset",
    )

    parser.add_argument(
        "--min_bin_length",
        type=float,
        default=8.0,
        required=False,
        help="specify the mininum bin length"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/media/zliu/data12/dataset/KITTI/VSRD_Format/feedforward_bins",
        required=False,
        help="name of info pkl",
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir,exist_ok=True)
    
    kitti360_data_prep(args)
