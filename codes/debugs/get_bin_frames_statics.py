
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import sys
import pickle
from tqdm import tqdm


def load_bin_file(path):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def get_statistics(numbers):
    if not numbers:
        return None, None, None, None  # Handle empty list

    sorted_nums = sorted(numbers)
    n = len(sorted_nums)

    # Calculate median
    if n % 2 == 1:
        median = sorted_nums[n // 2]
    else:
        median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2

    # Calculate mean
    mean = sum(numbers) / n

    return min(numbers), max(numbers), median, mean


if __name__=="__main__":
    
    min_frame_nums_threshold = 10
    root_path = "/data1/StereoDatasets/KITTI/KITTI360/feedforward_bins/"
    data_version = "bin_infos_8.0"
    #feedforward_bins
    all_files_path = "/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/trainval/train_2013_05_28_drive_0000_sync.txt"
    bin_tokens_list = read_text_lines(all_files_path)

    saved_path = "/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/more_sup_trainval/train_2013_05_28_drive_0000_sync.txt"

    valid_train_tokens_bigger_than_threshold = []
    
    for bin_name in tqdm(bin_tokens_list):
        abs_bin_name = os.path.join(root_path,data_version,bin_name)
        assert os.path.exists(abs_bin_name)
        bin_info_dicts = load_bin_file(abs_bin_name)
        
        bin_tokens_name = bin_info_dicts['token']
        cam_left_sensor_info_list = bin_info_dicts['sensor_info']['CAM_LEFT']
        
        if len(cam_left_sensor_info_list)>=min_frame_nums_threshold:
            valid_train_tokens_bigger_than_threshold.append(bin_name)
    
    
    with open(saved_path,'w') as f:
        for idx, fname in enumerate(valid_train_tokens_bigger_than_threshold):
            if idx!=len(valid_train_tokens_bigger_than_threshold)-1:
                f.writelines(fname+"\n")
            else:
                f.writelines(fname)
    

        
        
    
    
            
    
    # min_frame_nums, max_frame_nums,med_frame_nums,mean_frame_nums = get_statistics(frame_nums_inside_one_bin)
    
    # print("Min Frame Num: ",min_frame_nums)
    # print("Max Frame Num: ", max_frame_nums)
    # print("Med Frame Num: ",med_frame_nums)
    # print("Mean Frame Num: ",mean_frame_nums)