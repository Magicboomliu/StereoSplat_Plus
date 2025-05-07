import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import re
import sys

from kitti360scripts.helpers.project import CameraPerspective
from projection.lidar import loadVelodyneData,Kitti360Viewer3DRaw
from projection.rotation import rotation_matrix_x,expand_to_4x4
import skimage.io


import skimage.io
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
    
    


def kitti360_data_prep(args):
    
    # root path
    root_folder = args.root_folder
    
    data_2d_raw_folder = os.path.join(root_folder,"data_2d_raw")
    
    sequence_fname_list = os.listdir(data_2d_raw_folder)

    for seq_name in sorted(sequence_fname_list):        
        left_image_folder = os.path.join(data_2d_raw_folder,seq_name,"image_00","data_rect")

        for image_name in tqdm(sorted(os.listdir(left_image_folder))):
            
            # left image and right image path
            left_image_path = os.path.join(left_image_folder,image_name)
            right_image_path = left_image_path.replace("image_00","image_01")
            assert os.path.exists(left_image_path)
            assert os.path.exists(right_image_path)
            
            # calibration
            calibration_path = os.path.join(root_folder,"calibration/perspective.txt")
            assert os.path.exists(calibration_path)
            
            # velo lidar path
            velodyne_data_path = os.path.join(root_folder,'data_3d_raw',seq_name,'velodyne_points','data',os.path.basename(left_image_path).replace(".png",".bin"))
            assert os.path.exists(velodyne_data_path)
            
            sequence_id, instance_id = get_seq_id_and_instance_id(path=left_image_path)
            
            
            # left cam and right cam
            camera_left = CameraPerspective(root_folder, sequence_id, 0) # K,R, height, width: get all the intrincs
            camera_right = CameraPerspective(root_folder, sequence_id, 1) # K,R, height, width: get all the intrincs

            # velo
            seq_number = get_sequence_frame_number(sequence_id)
            velo_data = Kitti360Viewer3DRaw(mode='velodyne', seq=seq_number,kitti360_path=root_folder)

                    
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
            
            
            sparse_depth_map_left_uint16 = save_results_into_uint16(results=sparse_depth_map_left,scale_factor=256)
            sparse_depth_map_right_uint16 = save_results_into_uint16(results=sparse_depth_map_right,scale_factor=256)
            
            saved_sparse_depth_map_path_left = left_image_path.replace(args.root_folder,args.output_folder)
            saved_sparse_depth_map_path_right = right_image_path.replace(args.root_folder,args.output_folder)
            
            os.makedirs(os.path.dirname(saved_sparse_depth_map_path_left),exist_ok=True)
            os.makedirs(os.path.dirname(saved_sparse_depth_map_path_right),exist_ok=True)

            skimage.io.imsave(saved_sparse_depth_map_path_left,sparse_depth_map_left_uint16)
            skimage.io.imsave(saved_sparse_depth_map_path_right,sparse_depth_map_right_uint16)
            


    
    
if __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Data converter arg parser")

    parser.add_argument(
        "--root_folder",
        type=str,
        required=False,
        default="/media/zliu/data12/dataset/KITTI/VSRD_Format/",
        help="specify the root path of dataset",
    )
    
    parser.add_argument(
        "--output_folder",
        type=str,
        required=False,
        default="/media/zliu/data12/dataset/KITTI/VSRD_Format/",
        help="specify the root path of dataset",
    )


    args = parser.parse_args()
    os.makedirs(args.output_folder,exist_ok=True)
    
    
    kitti360_data_prep(args)
