import torch
import os
import numpy as np
import sys
import json

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__=="__main__":
    root_path = "/data1/StereoDatasets/KITTI/KITTI360/"
    sequence_name_path = "/home/zliu/Project2025/FeedStereoGS/Step2FusionCodes/filenames/raw_filenames/2013_05_28_drive_0000_sync_list.txt"   
    
    framename_list = sorted(read_text_lines(sequence_name_path))
    
    for idx,fname in enumerate(framename_list):
        annotaions_path = os.path.join(root_path,fname)
        annotation_info = read_text_lines(annotaions_path)
        left_cam_to_world_pose = annotation_info['left_cam_to_lidar']
        