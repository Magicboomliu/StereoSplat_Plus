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

# import sys
# sys.path.append("../..")
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

        # input_c2ws = torch.as_tensor(input_c2ws, dtype=torch.float32) #(2,4,4)----> Center Frame
        input_c2ws = np.array(input_c2ws)  # 变成 shape=(2, 4, 4) 的 ndarray
        input_c2ws = torch.from_numpy(input_c2ws).float()  # 再转成 tensor
        
        input_w2cs = np.array(input_w2cs)  # 变成统一的 (2, 4, 4) ndarray
        input_w2cs = torch.from_numpy(input_w2cs).float()  # 更快、更标准
        

        input_imgs, input_depths, input_depths_m, input_confs_m, input_cks = \
                    load_conditions(input_img_paths, self.reso)      

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
        
        # prepare w2i for volume-gs---> World 2 Images
        input_w2is = []
        for w2c, ck in zip(input_w2cs, input_cks):
            viewpad = torch.eye(4)
            viewpad[:ck.shape[0], :ck.shape[1]] = ck
            w2i = (viewpad @ w2c.T)
            input_w2is.append(w2i)
        input_w2is = torch.stack(input_w2is) #(2,4,4), here is All Center
        
        # ======= Render views from non-key frames for rendering losses ====== #
        output_img_paths, output_c2ws, output_w2cs = [], [], []
        frame_num = len(bin_info["sensor_info"]["LIDAR_TOP"]) # how many frames, if no problems, here should be 7
        assert frame_num >= 3, "only got {} frames for bin{}".format(frame_num, bin_token)
        
        if self.use_center:
            rend_indices = [[1, 2]] * len(self.camera_types) # [[1, 2], [1, 2]]--> Frist and the Last
        else:
            rend_indices = [[0]] * len(self.camera_types)
        
        
        for cam_id, cam in enumerate(self.camera_types):
            indices = rend_indices[cam_id]
            for ind in indices:
                info = copy.deepcopy(bin_info["sensor_info"][cam][ind])
                img_path, c2w, w2c = load_info(info)
                img_path = os.path.join(self.datapath,img_path)
                output_img_paths.append(img_path)
                output_c2ws.append(c2w)
                output_w2cs.append(w2c)
        output_c2ws = torch.as_tensor(output_c2ws, dtype=torch.float32)  #(4,4,4)-->(2*6)
        
        
        # load and modify images (cropped or resized if necessary), and modify intrinsics accordingly
        output_imgs, output_depths, output_depths_m, output_confs_m, output_cks = \
                    load_conditions(output_img_paths, self.reso)
        output_fxs, output_fys, output_cxs, output_cys = output_cks[:, 0, 0], output_cks[:, 1, 1], output_cks[:, 0, 2], output_cks[:, 1, 2]
        
        # compute image fovs and pixel directions
        output_fovxs, output_fovys = [], []
        for fx, fy, cx, cy in zip(output_fxs, output_fys, output_cxs, output_cys):
            fovx = 2 * np.arctan(cx / fx)
            fovy = 2 * np.arctan(cy / fy)
            output_fovxs.append(fovx)
            output_fovys.append(fovy)
        output_fovxs = torch.as_tensor(output_fovxs, dtype=torch.float32)
        output_fovys = torch.as_tensor(output_fovys, dtype=torch.float32)


        # add input data to output
        output_imgs = torch.cat([output_imgs, input_imgs], dim=0)
        output_depths = torch.cat([output_depths, input_depths], dim=0)
        output_depths_m = torch.cat([output_depths_m, input_depths_m], dim=0)
        output_confs_m = torch.cat([output_confs_m, input_confs_m], dim=0)
        output_c2ws = torch.cat([output_c2ws, input_c2ws], dim=0) # first 2 dimension is the novel final ,final dimension is the input view
        output_fovxs = torch.cat([output_fovxs, input_fovxs], dim=0)
        output_fovys = torch.cat([output_fovys, input_fovys], dim=0)
        output_fxs = torch.cat([output_fxs, input_fxs], dim=0)
        output_fys = torch.cat([output_fys, input_fys], dim=0)
        output_cxs = torch.cat([output_cxs, input_cxs], dim=0)
        output_cys = torch.cat([output_cys, input_cys], dim=0)
        output_directions = []
        for fx, fy, cx, cy in zip(output_fxs, output_fys, output_cxs, output_cys):
            fovx = 2 * np.arctan(cx / fx)
            fovy = 2 * np.arctan(cy / fy)
            direction = get_ray_directions(self.reso[0], self.reso[1],
                                           focal=[fx, fy], principal=[cx, cy])
            output_directions.append(direction)
        output_directions = torch.stack(output_directions)
        output_rays_o, output_rays_d = get_rays(
                    output_directions, output_c2ws, keepdim=True, normalize=False)
        

        # pack data
        input_dict = {"rgb": input_imgs} # images
        input_dict_pix = {"depth_m": input_depths_m, "conf_m": input_confs_m,
                          "ck": input_cks, "c2w": input_c2ws,
                          "cx": input_cxs, "cy": input_cys, "fx": input_fxs, "fy": input_fys,
                          "rays_o": input_rays_o, "rays_d": input_rays_d} 
        
        input_dict_vol = {"w2i": input_w2is} # volume based methods

        output_dict = {"rgb": output_imgs, "depth": output_depths,
                       "depth_m": output_depths_m, "conf_m": output_confs_m,
                       "c2w": output_c2ws, "fovx": output_fovxs, "fovy": output_fovys,
                       "rays_o": output_rays_o, "rays_d": output_rays_d,
                       "input_image_path":output_img_paths+ input_img_paths
                       }

        return {
            "bin_token": bin_token_name,
            "outputs": output_dict,
            "inputs": input_dict,
            "inputs_pix": input_dict_pix,
            "inputs_vol": input_dict_vol
        }
    
    
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
        "resolution":[224, 840], # idx 0 is the proceseed image resolution, the last is the the initial image resolution
        "split":"train",
        "sequence":'2013_05_28_drive_0000_sync',
        "use_center":True,
        "use_first": False,
        "use_last": False,
    }
    
    dataset = KITTI360Dataset(**dataset_params)
    
    for idx, data in enumerate(dataset):
        print(data.keys())
        quit()