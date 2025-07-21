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
from .utils import read_text_lines

from .geometry_camera.image import resize_image,HWC3
from .geometry_camera.typing import *
from .geometry_camera.camera import get_camera,rescale_intrisic
from .geometry_camera.ops import get_cam_info_gaussian,get_ray_directions,get_rays

from .transforms.loading import get_inputs_info
import json

class KITTI360DatasetSequential(Dataset):    
    def __init__(
        self,
        datapath:str,
        train_filelist:str,
        val_filelist:str,
        test_filelist:str,
        sequence:Optional[str],
        resolution: list,
        split: str = "train",
        depth_info_dict: dict=None,
        camera_model: str='OpenCV',
        **kwargs,
        ):
        super().__init__()

        self.datapath = datapath
        self.sequence = sequence
        self.depth_info_dict = depth_info_dict
        self.camera_model = camera_model
        self.reso = resolution
        
        if split=='train':
            self.framename_list = read_text_lines(train_filelist)
        elif split=='val':
            self.framename_list = read_text_lines(val_filelist)
        elif split=='test':
            self.framename_list = read_text_lines(test_filelist)
        else:
            raise NotImplementedError

        self.split = split
        
        # get the the frame index
        all_avaiable_frame_idx = []
        for file in self.framename_list:
            all_avaiable_frame_idx.append(self._get_frame_index(file))
        all_avaiable_frame_idx = sorted(all_avaiable_frame_idx)
        
    
    def _get_frame_index(self,filename):
        return int(os.path.basename(filename[:-5]))

    

    def __getitem__(self, index):
        
        current_simple_annotation_path = self.framename_list[index]
        
        input_infos = get_inputs_info(datapath=self.datapath,
                        reso = self.reso,
                        simple_annotation_path_list=[current_simple_annotation_path],
                        depth_info_params = self.depth_info_dict,
                        extra_list=[])
        
        if self.split=='val':
            pass
            
        
        
                
        # =================== Input views of this bin ===================== #
        input_img_paths, input_c2ws, input_w2cs = [], [], []
        
        # loading the all information           
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
        
        # prepare w2i for volume-gs---> World 2 Images
        input_w2is = []
        for w2c, ck in zip(input_w2cs, input_cks):
            viewpad = torch.eye(4)
            viewpad[:ck.shape[0], :ck.shape[1]] = ck
            w2i = (viewpad @ w2c.T)
            input_w2is.append(w2i)
        input_w2is = torch.stack(input_w2is) #(2,4,4), here is All Center
        
        
        input_dict = {"rgb": input_imgs} # images
        input_dict_pix = {"ck": input_cks, "c2w": input_c2ws,
                        "cx": input_cxs, "cy": input_cys, "fx": input_fxs, "fy": input_fys,
                        "rays_o": input_rays_o, "rays_d": input_rays_d}
        

        
        
        
        input_imgs_list = torch.chunk(input_imgs,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_cks_list = torch.chunk(input_cks,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_c2ws_list = torch.chunk(input_c2ws,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_cxs_list = torch.chunk(input_cxs,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_fxs_list = torch.chunk(input_fxs,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_fys_list = torch.chunk(input_fys,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_rays_o_list = torch.chunk(input_rays_o,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_rays_d_list = torch.chunk(input_rays_d,chunks=len(input_imgs)//self.pair_images,dim=0)
        
        input_fovxs_list = torch.chunk(input_fovxs,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_fovys_list = torch.chunk(input_fovys,chunks=len(input_imgs)//self.pair_images,dim=0)
        
        input_depths_m_list = torch.chunk(input_depths_m,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_depths_list = torch.chunk(input_depths,chunks=len(input_imgs)//self.pair_images,dim=0)
        input_confs_m_list = torch.chunk(input_confs_m,chunks=len(input_imgs)//self.pair_images,dim=0)
    
        
        
        input_sparse_gt_depth_list = torch.chunk(input_sparse_gt_depth,chunks=len(input_imgs)//self.pair_images,dim=0)

        input_w2is_list = torch.chunk(input_w2is,chunks=len(input_imgs)//self.pair_images,dim=0)
        
        
        input_img_paths_list = self.split_list_into_pairs(input_img_paths)
        
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
                          "cx": input_cxs_list, "cy": input_cys, 
                          "fx": input_fxs_list, "fy": input_fys_list,
                          "rays_o": input_rays_o_list, "rays_d": input_rays_d_list}

        
        if self.depth_info_dict.use_pseudo_depth:
            input_dict_pix.update({
                                    "depth_m":input_depths_m_list,
                                    "conf_m":input_confs_m_list
                                    })
            
        if self.depth_info_dict.use_sparse_lidar:
            input_dict_pix.update({
                                    "sparse_gt_depth":input_sparse_gt_depth_list
                                    })
        
        input_dict_vol = {"w2i": input_w2is_list} # volume based methods

    
        # if self.split=='train':
        output_dict = {"rgb": self.swap_first_three(input_imgs_list), 
                    "c2w": self.swap_first_three(input_c2ws_list), 
                    "fovx": self.swap_first_three(input_fovxs_list), 
                    "fovy": self.swap_first_three(input_fovys_list),
                    "rays_o": self.swap_first_three(input_rays_o_list), 
                    "rays_d": self.swap_first_three(input_rays_d_list),
                    "input_image_path":self.swap_first_three(input_img_paths_list)
                    }
        
        if self.depth_info_dict.use_pseudo_depth:
            output_dict.update({
                "depth": self.swap_first_three(input_depths_list),
                    "depth_m": self.swap_first_three(input_depths_m_list), 
                    "conf_m": self.swap_first_three(input_confs_m_list)
            })
        if self.depth_info_dict.use_sparse_lidar:
            output_dict.update({
                'sparse_gt_depth':self.swap_first_three(input_sparse_gt_depth_list),
            })
                
  
        return {
            "bin_token": bin_token_name,
            "bin_filenames": input_img_paths,
            "outputs": output_dict,
            "inputs": input_dict,
            "inputs_pix": input_dict_pix,
            "inputs_vol": input_dict_vol
        }
    
    


    def __len__(self):
        return len(self.framename_list)


if __name__=="__main__":
    
    
    pass



