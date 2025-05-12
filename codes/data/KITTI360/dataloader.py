import os
import os.path as osp
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import json
import random
import pickle as pkl
from functools import cached_property
from pathlib import Path
import imageio.v2 as imageio
import glob
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import cv2
import copy
from io import BytesIO
from einops import rearrange, repeat, einsum
cv2.setNumThreads(0) 
cv2.ocl.setUseOpenCL(False)

import sys
sys.path.append("../..")
from model.utils.image import resize_image, HWC3
from model.utils.typing import *
from model.utils.camera import get_camera, rescale_intrisic
from model.utils.ops import get_cam_info_gaussian, get_ray_directions, get_rays
from data.KITTI360.transforms.loading import load_info, load_conditions


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

class KITTI360Dataset(Dataset):
    
    def __init__(
        self,
        datapath:str,
        train_filelist:str,
        val_filelist:str,
        test_filelist:str,
        data_version:str,
        resolution: list,
        split: str = "train",
        sequence:str ='2013_05_28_drive_0000_sync',
        use_center: bool = True,
        use_first: bool = False,
        use_last: bool = False,
        **kwargs,
        ):
        super().__init__()

        self.datapath = datapath
        self.data_version = data_version
        
        self.camera_types = [
            "CAM_LEFT",
            "CAM_RIGHT"
        ]

        self.camera_types_first = [
            "CAM_LEFT",
            "CAM_RIGHT"
        ]
        self.camera_types_last = [
            "CAM_LEFT",
            "CAM_RIGHT"
        ]
        
        self.reso = resolution
        self.resize_reso = resolution[0]
        self.original_reso = resolution[1]
        # using the center as the input
        self.use_center = use_center
        
        # using the last or first for input， or just for validations
        self.use_first = use_first
        self.use_last = use_last
        
        if split =='train':
            self.bin_tokens = read_text_lines(train_filelist)
        
        elif split =="val":
            self.bin_tokens = read_text_lines(val_filelist)
        
        elif split =='test':
            self.bin_tokens = read_text_lines(val_filelist)
        
        elif split =='demo':
            raise NotImplementedError
            
        self.split = split

    def __getitem__(self, index):
        
        bin_token_name = self.bin_tokens[index]
        abs_bin_token_fname = os.path.join(self.datapath,"feedforward_bins",self.data_version,bin_token_name)
        
        bin_info = self._load_pkl_file(abs_bin_token_fname)
        # print(bin_info['token']) # scene2013_05_28_drive_0000_sync_bin000 
        # print(bin_info['scene_token']) # 2013_05_28_drive_0000_sync
        # print(bin_info['timestep']) # 0000000256.
        # print(bin_info['bin_length']) # 8.402467621701142
        # print(bin_info['sensor_info'].keys()) # dict_keys(['LIDAR_TOP', 'CAM_LEFT', 'CAM_RIGHT'])
        
        # center
        sensor_info_center = {sensor: bin_info["sensor_info"][sensor][0] for sensor in self.camera_types + ["LIDAR_TOP"]}
        # first 
        sensor_info_first = {sensor: bin_info["sensor_info"][sensor][1] for sensor in self.camera_types_first + ["LIDAR_TOP"]}
        # last
        sensor_info_last = {sensor: bin_info["sensor_info"][sensor][2] for sensor in self.camera_types_last + ["LIDAR_TOP"]}

        # =================== Input views of this bin ===================== #
        input_img_paths, input_c2ws, input_w2cs = [], [], []
        if self.use_center:
            for cam in self.camera_types:

                info = copy.deepcopy(sensor_info_center[cam]) # all the infors
                img_path, c2w, w2c = load_info(info)
                img_path = os.path.join(self.datapath,img_path)
                assert os.path.exists(img_path)
                input_img_paths.append(img_path)
                input_c2ws.append(c2w)
                input_w2cs.append(w2c)

        if self.use_first:
            for cam in self.camera_types_first:
                info = copy.deepcopy(sensor_info_first[cam])
                img_path, c2w, w2c = load_info(info)
                img_path = os.path.join(self.datapath,img_path)
                input_img_paths.append(img_path)
                input_c2ws.append(c2w)
                input_w2cs.append(w2c)
                
        if self.use_last:
            for cam in self.camera_types_last:
                info = copy.deepcopy(sensor_info_last[cam])
                img_path, c2w, w2c = load_info(info)
                img_path = os.path.join(self.datapath,img_path)
                input_img_paths.append(img_path)
                input_c2ws.append(c2w)
                input_w2cs.append(w2c)

        input_c2ws = torch.as_tensor(input_c2ws, dtype=torch.float32) #(2,4,4)----> Center Frame
        input_w2cs = torch.as_tensor(input_w2cs, dtype=torch.float32) #(2,4,4)----> Center Frame


        input_imgs, input_depths, input_depths_m, input_confs_m, input_cks = \
                    load_conditions(input_img_paths, self.reso)      

        input_cks = torch.as_tensor(input_cks, dtype=torch.float32) #(6,3,3)--> Camera Intrinsics
        
        

        

        
        

    def _load_pkl_file(self,path):
        with open(path, 'rb') as f:
            data_dict = pkl.load(f)
        return data_dict


    def __len__(self):
        return len(self.bin_tokens)



if __name__=="__main__":
    
    dataset_params = {
        "datapath":"/data1/StereoDatasets/KITTI/KITTI360/",
        "train_filelist":"/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/train_2013_05_28_drive_0000_sync.txt",
        "val_filelist":"/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt",
        "test_filelist":"/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt",
        "data_version":"bin_infos_8.0",
        "resolution":[[224, 840],[376,1408]], # idx 0 is the proceseed image resolution, the last is the the initial image resolution
        "split":"train",
        "sequence":'2013_05_28_drive_0000_sync',
        "use_center":True,
        "use_first": False,
        "use_last": False,
    }
    
    dataset = KITTI360Dataset(**dataset_params)
    
    for idx, data in enumerate(dataset):
        print(data)