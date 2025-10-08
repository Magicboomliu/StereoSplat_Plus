import os
import json
import pickle as pkl
import numpy as np
from tqdm import tqdm 
import mmengine

def load_pkl_file(path):
    with open(path, 'rb') as f:
        data_dict = pkl.load(f)
    return data_dict


def convert_coordinate_from_center_lidar_to_first_lidar(normal_bin_data):
    
    first_frame_lidar_pose = normal_bin_data['sensor_info']['LIDAR_TOP'][1]['sensor2lidar_transform']
    center_frame_lidar_pose = normal_bin_data['sensor_info']['LIDAR_TOP'][0]['sensor2lidar_transform']
    center_to_first_lidar_pose = np.linalg.inv(first_frame_lidar_pose) @ center_frame_lidar_pose
    
    for sensor_type in normal_bin_data['sensor_info'].keys():
        for frame_ind in range(len(normal_bin_data['sensor_info'][sensor_type])):
            normal_bin_data['sensor_info'][sensor_type][frame_ind]['sensor2lidar_transform'] = center_to_first_lidar_pose @ normal_bin_data['sensor_info'][sensor_type][frame_ind]['sensor2lidar_transform'] 
            normal_bin_data['sensor_info'][sensor_type][frame_ind]['sensor2lidar_rotation'] = normal_bin_data['sensor_info'][sensor_type][frame_ind]['sensor2lidar_transform'][:3,:3]
            normal_bin_data['sensor_info'][sensor_type][frame_ind]['sensor2lidar_translation'] = normal_bin_data['sensor_info'][sensor_type][frame_ind]['sensor2lidar_transform'][:3,3]
    
    return normal_bin_data


if __name__=="__main__":
    
    normal_bin_path_folder = "/data1/StereoDatasets/KITTI/KITTI360/feedforward_bins/bin_infos_8.0"
    save_bin_path_folder = "/data1/StereoDatasets/KITTI/KITTI360/feedforward_bins_completed/bin_infos_8.0_FirstLIDAR"
        
    for fname in tqdm(sorted(os.listdir(normal_bin_path_folder))):
        
        normal_bin_path = os.path.join(normal_bin_path_folder,fname)
        normal_bin_data = load_pkl_file(normal_bin_path)
        normal_bin_data = convert_coordinate_from_center_lidar_to_first_lidar(normal_bin_data)
        
        
        new_saved_bin_path = os.path.join(save_bin_path_folder,fname)
        
        
        mmengine.dump(normal_bin_data, new_saved_bin_path)

        
        

