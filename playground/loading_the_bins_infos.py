import os
import pickle as pkl
import numpy as np 

def load_pkl_file(path):
    with open(path, 'rb') as f:
        data_dict = pkl.load(f)
    return data_dict




if __name__=="__main__":
    bin_path = "/data1/StereoDatasets/KITTI/KITTI360/feedforward_bins/bin_infos_8.0_FirstLIDAR/scene2013_05_28_drive_0000_sync_bin000.pkl"
    assert os.path.exists(bin_path)
    bin_infos_data =load_pkl_file(bin_path)
    
    
    left_cam_pose = bin_infos_data['sensor_info']['CAM_LEFT'][5]['sensor2lidar_transform']
    right_cam_pose = bin_infos_data['sensor_info']['CAM_RIGHT'][5]['sensor2lidar_transform']
    
    print(np.linalg.inv(left_cam_pose) @ right_cam_pose)

