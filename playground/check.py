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
# filename = "example_file.txt"
# name_without_ext = Path(filename).stem
import re
import open3d as o3d


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



if __name__=="__main__":
    
    root_path = "/media/zliu/data12/dataset/KITTI/VSRD_Format/"
    
    bin_path = "/media/zliu/data12/dataset/KITTI/VSRD_Format/feedforward_bins/bin_infos/scene2013_05_28_drive_0000_sync_bin031.pkl"

    with open(bin_path, "rb") as f:
        bin_info = pkl.load(f)
        
    print(bin_info['token'])
    print(bin_info['scene_token'])
    print(bin_info['bin_length'])
    print(bin_info['sensor_info'].keys()) # dict_keys(['LIDAR_TOP', 'CAM_LEFT', 'CAM_RIGHT'])
    
    center_frame_sensor_lidar = bin_info['sensor_info']['LIDAR_TOP'][0]
    center_frame_sensor_left = bin_info['sensor_info']['CAM_LEFT'][0]
    center_frame_sensor_right = bin_info['sensor_info']['CAM_RIGHT'][0]

    # print(center_frame_sensor_lidar['data_path'])
    # print(center_frame_sensor_left['data_path'])
    # print(center_frame_sensor_right['data_path'])
    
    seq_name = get_sequence_frame_number(get_sequence_name(path=center_frame_sensor_lidar['data_path']))
    velo = Kitti360Viewer3DRaw(mode='velodyne', seq=seq_name,kitti360_path=root_path)
    instance_id = int(Path(os.path.basename(center_frame_sensor_lidar['data_path'])).stem)
    
    points = velo.loadVelodyneData(instance_id)
    points = velo.curlVelodyneData(instance_id, points)
    points[:,3] = 1
    

    
    visualize_point_cloud_with_axis(points[:,:3])

    
    
    
    

    



