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
from types import SimpleNamespace
import sys
sys.path.append("../..")
from stereosplat.model.utils.image import resize_image, HWC3
from stereosplat.model.utils.typing import *
from stereosplat.model.utils.camera import get_camera, rescale_intrisic
from stereosplat.model.utils.ops import get_cam_info_gaussian, get_ray_directions, get_rays
from stereosplat.data.KITTI360_VGGT.transforms.loading import load_info,load_conditions
# from data.KITTI360.transforms.loading import load_info,load_conditions
import matplotlib.pyplot as plt
import random


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def swap_elements(lst, A, B):
    lst[A], lst[B] = lst[B], lst[A]

class KITTI360DatasetVGGT(Dataset):    
    def __init__(
        self,
        datapath:str,
        train_filelist:str,
        val_filelist:str,
        data_version:str,
        resolution: list,
        split: str = "train",
        depth_info_dict: dict=None,
        camera_model: str='OpenGL',
        input_type:str='all',
        max_input_views:int=10,
        pair_images:int=2,
        names_of_frames:int=2,
        **kwargs,
        ):
        super().__init__()

        self.datapath = datapath
        self.data_version = data_version
        self.depth_info_dict = depth_info_dict
        self.camera_model = camera_model
        
        self.input_type = input_type
        self.max_input_views = max_input_views
        
        self.pair_images = pair_images
        
        self.names_of_frames = names_of_frames
        
        
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

        
        if split =='train':
            self.bin_tokens = read_text_lines(train_filelist)
        
        elif split =="val":
            self.bin_tokens = read_text_lines(val_filelist)
        
        elif split =='test':
            self.bin_tokens = read_text_lines(val_filelist)
        
        elif split =='demo':
            raise NotImplementedError
            
        self.split = split
        

    def _uniform_sample(self,ordered_list, N):
        if N <= 0 or N > len(ordered_list):
            raise ValueError("N must be > 0 and <= length of ordered_list")

        total = len(ordered_list)
        step = (total - 1) / (N - 1) if N > 1 else 0
        indices = []
        last_index = -1
        for i in range(N):
            idx = round(i * step)
            if idx == last_index:
                idx += 1  # 确保不重复（尽量）
            idx = min(idx, total - 1)  # 防止越界
            indices.append(idx)
            last_index = idx
        return [ordered_list[i] for i in indices]

    def __getitem__(self, index):
        
        bin_token_name = self.bin_tokens[index]
        abs_bin_token_fname = os.path.join(self.datapath,"feedforward_bins",self.data_version,bin_token_name)    
        bin_info = self._load_pkl_file(abs_bin_token_fname) # dict_keys(['token', 'scene_token', 'timestep', 'bin_length', 'sensor_info'])
        
        '''dict_keys(['LIDAR_TOP', 'CAM_LEFT', 'CAM_RIGHT'])'''
        
        # rest
        sensor_info_all = [{sensor: bin_info["sensor_info"][sensor][i] for sensor in self.camera_types_last + ["LIDAR_TOP"]}
                            for i in range(len(bin_info['sensor_info'][self.camera_types[0]]))]

        # N Views
        nums_of_all_view = len(sensor_info_all)
        selected_index = sorted(random.sample(list(range(nums_of_all_view)), self.names_of_frames))
        sensor_info_all = [sensor_info_all[ind] for ind in selected_index]
        
        # =================== Input views of this bin ===================== #
        input_img_paths, input_c2ws, input_w2cs = [], [], []
        

        if self.input_type=='all':
            
            for sensor_info in sensor_info_all:
                for cam in self.camera_types:
                    info = copy.deepcopy(sensor_info[cam]) # all the info
                    img_path, c2w, w2c = load_info(info,cam_type=self.camera_model)
                    img_path = os.path.join(self.datapath,img_path)
                    assert os.path.exists(img_path)
                    input_img_paths.append(img_path)
                    input_c2ws.append(c2w)
                    input_w2cs.append(w2c)

            # input_c2ws = torch.as_tensor(input_c2ws, dtype=torch.float32) #(2,4,4)----> Center Frame
            input_c2ws = np.array(input_c2ws)  # 变成 shape=(2, 4, 4) 的 ndarray
            input_c2ws = torch.from_numpy(input_c2ws).float()  # 再转成 tensor
            
            input_w2cs = np.array(input_w2cs)  # 变成统一的 (2, 4, 4) ndarray
            input_w2cs = torch.from_numpy(input_w2cs).float()  # 更快、更标准
        
            input_imgs, input_cks, input_depths_dict = load_conditions(img_paths=input_img_paths,
                                                                    reso=self.reso,
                                                                    depth_info_params=self.depth_info_dict)
            if self.depth_info_dict.use_pseudo_depth:
                input_depths = input_depths_dict['depths']
                input_depths_m = input_depths_dict['depths_m']
                input_confs_m = input_depths_dict['confs_m']
            
            if self.depth_info_dict.use_sparse_lidar:
                input_sparse_gt_depth = input_depths_dict['sparse_gts']
            
            input_cks = torch.as_tensor(input_cks, dtype=torch.float32) #(6,3,3)--> Camera Intrinsics
            # get the fx, fy, cx,cy 
            input_fxs, input_fys, input_cxs, input_cys = input_cks[:, 0, 0], input_cks[:, 1, 1], input_cks[:, 0, 2], input_cks[:, 1, 2]
            
            # compute image fovs and pixel directions
            input_fovxs, input_fovys = [], []
            input_directions = []

            # https://blog.csdn.net/OrdinaryMatthew/article/details/126670351
            for fx, fy, cx, cy in zip(input_fxs, input_fys, input_cxs, input_cys):
                direction = get_ray_directions(self.reso[0], self.reso[1],
                                            focal=[fx, fy], principal=[cx, cy]) # openGL
                fovx = 2 * np.arctan(cx / fx)
                fovy = 2 * np.arctan(cy / fy)
                input_fovxs.append(fovx)
                input_fovys.append(fovy)
                input_directions.append(direction)
                
            input_fovxs = torch.as_tensor(input_fovxs, dtype=torch.float32) #(6)
            input_fovys = torch.as_tensor(input_fovys, dtype=torch.float32) #(6)
            input_directions = torch.stack(input_directions) #(6,H,W,3)

            # shape is [2,H,W,3]
            # shape is [2,H,W,3]
            input_rays_o, input_rays_d = get_rays(
                input_directions, input_c2ws, keepdim=True, normalize=False)
            

            
            
            input_dict = {"rgb": input_imgs} # images
            input_dict_pix = {"ck": input_cks, "c2w": input_c2ws,
                            "cx": input_cxs, "cy": input_cys, "fx": input_fxs, "fy": input_fys,
                            "rays_o": input_rays_o, "rays_d": input_rays_d}



        input_imgs_list = torch.chunk(input_imgs,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_cks_list = torch.chunk(input_cks,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_c2ws_list = torch.chunk(input_c2ws,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_cxs_list = torch.chunk(input_cxs,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_cys_list = torch.chunk(input_cys,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_fxs_list = torch.chunk(input_fxs,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_fys_list = torch.chunk(input_fys,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_rays_o_list = torch.chunk(input_rays_o,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_rays_d_list = torch.chunk(input_rays_d,chunks=len(input_imgs)//self.pair_images,dim=0)
        
        input_depths_m_list = torch.chunk(input_depths_m,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_confs_m_list = torch.chunk(input_confs_m,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_sparse_gt_depth_list = torch.chunk(input_sparse_gt_depth,chunks=len(input_imgs)//self.pair_images,dim=0)


        
        
        input_rgb_names_list = []
        current_rgb_list =[]
        for idx, img_path in enumerate(input_img_paths):
            current_rgb_list.append(img_path)
            if len(current_rgb_list)%self.pair_images==0:
                input_rgb_names_list.append(current_rgb_list)
                current_rgb_list = []

            
        # pack data
        input_dict = {"rgb": input_imgs_list,
                      "rgb_path": input_rgb_names_list,
                      } # images

        input_dict_pix = {"ck": input_cks_list, "c2w": input_c2ws_list,
                          "cx": input_cxs_list, "cy": input_cys_list, "fx": input_fxs_list, "fy": input_fys_list,
                          "rays_o": input_rays_o_list, "rays_d": input_rays_d_list}

        
        if self.depth_info_dict.use_pseudo_depth:
            input_dict_pix.update({"depth_m":input_depths_m_list,
                                    "conf_m":input_confs_m_list
                                    })
            
        if self.depth_info_dict.use_sparse_lidar:
            input_dict_pix.update({
                                    "sparse_gt_depth":input_sparse_gt_depth_list
                                    })

  
        return {
            "bin_token": bin_token_name,
            "bin_filenames": input_img_paths,
            "inputs": input_dict,
            "inputs_pix": input_dict_pix}
    
    
    def _load_pkl_file(self,path):
        with open(path, 'rb') as f:
            data_dict = pkl.load(f)
        return data_dict


    def __len__(self):
        return len(self.bin_tokens)



if __name__=="__main__":

    _REPO_ROOT = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..")
    )
    _FILENAMES = os.path.join(_REPO_ROOT, "filenames", "kitti360", "trainval")


    depth_info_params = dict(
        use_pseudo_depth=True,
        pseudo_depth_type='NMRFStereo', # select from "MonocularDepthV2", "Metric3DV2","NMRFStereo"
        use_sparse_lidar=True
        )
    
    ns = SimpleNamespace(**depth_info_params)
    

    dataset_params = {
        "datapath":os.environ.get("KITTI360_DATAPATH", "/path/to/KITTI360"),
        "train_filelist":os.path.join(_FILENAMES, "train_2013_05_28_drive_0000_sync.txt"),
        "val_filelist":os.path.join(_FILENAMES, "val_2013_05_28_drive_0000_sync.txt"),
        "test_filelist":os.path.join(_FILENAMES, "val_2013_05_28_drive_0000_sync.txt"),
        "data_version":"bin_infos_8.0",
        "resolution":[112, 518], # idx 0 is the proceseed image resolution, the last is the the initial image resolution
        "split":"train",
        "sequence":'2013_05_28_drive_0000_sync',
        "camera_model":"OpenCV",
        "depth_info_dict":ns,
        "input_type": "all", # select from all, or "stereo" or "max"
        "max_input_views": 10,
        "pair_images":2
    }
    
    val_dataset = KITTI360DatasetVGGT(**dataset_params)
    

    val_dataloader = DataLoader(
        val_dataset, 1, shuffle=False,
        num_workers=0
    )
    

    
    for idx, data in enumerate(val_dataloader):
        
        print(len(data['inputs']['rgb']))
        print(data['inputs_pix']['depth_m'][0].shape)
        print(data['inputs_pix']['c2w'][0].shape)
        print(data['inputs_pix']['cx'][0].shape)
        print(data['inputs_pix']['cy'][0].shape)
        print(data['inputs_pix']['fx'][0].shape)
        print(data['inputs_pix']['fy'][0].shape)
        
        # nums_of_all_view = len(data['inputs']['rgb'])
        # selected_index = sorted(random.sample(list(range(nums_of_all_view)), 2))
        # print(selected_index)
        # print(nums_of_all_view)
        # print(data['inputs_pix'].keys())
        # print(data['inputs']['rgb'][0].shape)
        
        quit()