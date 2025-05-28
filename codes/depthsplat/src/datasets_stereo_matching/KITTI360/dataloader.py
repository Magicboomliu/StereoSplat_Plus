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
sys.path.append("../../../..")
from depthsplat.src.datasets_stereo_matching.KITTI360.transforms.loading import load_condiations

# debug here
import matplotlib.pyplot as plt


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


class KITTI360Dataset(Dataset):
    def __init__(self,
                 datapath,
                 train_filelist:str,
                 val_filelist:str,
                 test_filelist:str,
                 resolution: list,
                 split: str = "train",
                 use_projected_lidar:bool=True,
                 use_pseudo_depth: bool = True,
                **kwargs,
                 
                 ):
        super().__init__()
        
        self.datapath = datapath
        
        self.train_filelist = train_filelist
        self.val_filelist = val_filelist
        self.test_filelist = test_filelist
        self.resolution = resolution
        self.split = split
        self.use_projected_lidar = use_projected_lidar
        self.use_pseudo_depth = use_pseudo_depth
        
        if split =='train':
            self.filenames = read_text_lines(train_filelist)
        
        elif split =='val':
            self.filenames = read_text_lines(val_filelist)
        
        elif split =='test':
            self.filenames = read_text_lines(test_filelist)
        
    
    def __getitem__(self, index):
        
        sample_data = dict()
        
        current_filemname = self.filenames[index]
        basename = current_filemname.replace("/","@")[:-5]
        sample_data['filenames'] = [basename]

        imgs,cKs,cTs,depths_dict = load_condiations(annotation_path=current_filemname,reso=self.resolution,
                         datapath=self.datapath,use_projected_lidar=self.use_projected_lidar,
                         use_pseudo_depth=self.use_pseudo_depth)
        
        if 'sparse_depths' in depths_dict.keys():
            sparse_depths = depths_dict['sparse_depths']
        
        if 'pseudo_depths' in depths_dict.keys():
            pseudo_depths = depths_dict['pseudo_depths']
            
            
        sample_data['imgs'] = imgs
        sample_data['intrinsics'] = cKs
        sample_data['extrinsics'] = cTs
        
        if self.use_projected_lidar:
            sample_data['sparse_depths'] = sparse_depths
        
        if self.use_pseudo_depth:
            sample_data['pseudo_depths'] = pseudo_depths
        
        cameras_dist_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # [2, 2] [V,K]
        sample_data['nn_matrix'] = cameras_dist_index
        
        
        return sample_data

    def _load_pkl_file(self,path):
        with open(path, 'rb') as f:
            data_dict = pkl.load(f)
        return data_dict


    def __len__(self):
        return len(self.filenames)

    

def warp_image2_to_image1(image2, depth_map1, K, T):
    B, _, H, W = image2.shape

    K1 = K[:, 0]     # [B,3,3] cam1 intrinsics
    K2 = K[:, 1]
    T1 = T[:, 0]     # [B,4,4] cam1 cam2world
    T2 = T[:, 1]

    depth = depth_map1[:, 0]  # [B,H,W] — cam1 depth

    # meshgrid
    y, x = torch.meshgrid(
        torch.arange(H, device=image2.device),
        torch.arange(W, device=image2.device),
        indexing='ij'
    )
    ones = torch.ones_like(x)
    pix_coords = torch.stack((x, y, ones), dim=0).float()  # [3, H, W]
    pix_coords = pix_coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 3, H, W]

    # pixel to cam1 coordinates
    K1_inv = torch.inverse(K1)  # [B,3,3]
    cam1_coords = K1_inv @ pix_coords.view(B, 3, -1)  # [B,3,H*W]
    cam1_coords = cam1_coords * depth.view(B, 1, -1)  # scale by depth

    # cam1 to world
    cam1_coords_homo = torch.cat([cam1_coords, torch.ones((B, 1, H*W), device=image2.device)], dim=1)  # [B,4,N]
    world_coords = T1 @ cam1_coords_homo  # [B,4,N]

    # world to cam2
    T2_inv = torch.inverse(T2)  # world2cam2
    cam2_coords_homo = T2_inv @ world_coords  # [B,4,N]
    cam2_coords = cam2_coords_homo[:, :3] / cam2_coords_homo[:, 2:3]  # [B,3,N]

    # project to image2
    proj_coords = K2 @ cam2_coords  # [B,3,N]
    u = proj_coords[:, 0] / proj_coords[:, 2]  # [B,N]
    v = proj_coords[:, 1] / proj_coords[:, 2]

    # normalize to [-1,1] for grid_sample
    u_norm = (u / (W - 1)) * 2 - 1
    v_norm = (v / (H - 1)) * 2 - 1

    grid = torch.stack((u_norm, v_norm), dim=-1)  # [B,N,2]
    grid = grid.view(B, H, W, 2)

    # sample image2 at projected points
    warped_image2 = F.grid_sample(image2, grid, mode='bilinear', align_corners=True)

    return warped_image2



def warp_image1_to_image2(image1, depth_map2, K, T):
    B, _, H, W = image1.shape

    K1 = K[:, 0]     # [B,3,3] cam1 intrinsics
    K2 = K[:, 1]
    T1 = T[:, 0]     # [B,4,4] cam1 cam2world
    T2 = T[:, 1]

    depth = depth_map2[:, 0]  # [B,H,W] — cam1 depth

    # meshgrid
    y, x = torch.meshgrid(
        torch.arange(H, device=image1.device),
        torch.arange(W, device=image1.device),
        indexing='ij'
    )
    ones = torch.ones_like(x)
    pix_coords = torch.stack((x, y, ones), dim=0).float()  # [3, H, W]
    pix_coords = pix_coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 3, H, W]

    # pixel to cam1 coordinates
    K2_inv = torch.inverse(K2)  # [B,3,3]
    cam2_coords = K2_inv @ pix_coords.view(B, 3, -1)  # [B,3,H*W]
    cam2_coords = cam2_coords * depth.view(B, 1, -1)  # scale by depth

    # cam1 to world
    cam2_coords_homo = torch.cat([cam2_coords, torch.ones((B, 1, H*W), device=image1.device)], dim=1)  # [B,4,N]
    world_coords = T2 @ cam2_coords_homo  # [B,4,N]

    # world to cam2
    T1_inv = torch.inverse(T1)  # world2cam2
    cam1_coords_homo = T1_inv @ world_coords  # [B,4,N]
    cam1_coords = cam1_coords_homo[:, :3] / cam1_coords_homo[:, 2:3]  # [B,3,N]

    # project to image2
    proj_coords = K1 @ cam1_coords  # [B,3,N]
    u = proj_coords[:, 0] / proj_coords[:, 2]  # [B,N]
    v = proj_coords[:, 1] / proj_coords[:, 2]

    # normalize to [-1,1] for grid_sample
    u_norm = (u / (W - 1)) * 2 - 1
    v_norm = (v / (H - 1)) * 2 - 1

    grid = torch.stack((u_norm, v_norm), dim=-1)  # [B,N,2]
    grid = grid.view(B, H, W, 2)

    # sample image2 at projected points
    warped_image2 = F.grid_sample(image1, grid, mode='bilinear', align_corners=True)

    return warped_image2


if __name__=="__main__":
    
    datapath = "/data1/StereoDatasets/KITTI/KITTI360/"
    train_filelist = "/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/depth_estimation_trainval/train.txt"
    val_filelist = "/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/depth_estimation_trainval/val.txt"
    test_filelist = "/home/zliu/Project2025/Feedforward_Based_3DGS/more_supp_vanilla/FeedStereoGS/filenames/kitti360/depth_estimation_trainval/val.txt"
    resolution = [224,832]
    
    
    dataset_params = {
    "train": {
        "datapath": datapath,
        "train_filelist": train_filelist,
        "val_filelist": val_filelist,
        "test_filelist":test_filelist,
        "resolution": resolution,
        "split": "train",
        "use_projected_lidar":True,
        "use_pseudo_depth":True},
    
    "val":{
        "datapath": datapath,
        "train_filelist": train_filelist,
        "val_filelist": val_filelist,
        "test_filelist":test_filelist,
        "resolution": resolution,
        "split": "val",
        "use_projected_lidar":True,
        "use_pseudo_depth":True
        }
    }
    
    train_dataset = KITTI360Dataset(**dataset_params["train"])
    
    val_dataset = KITTI360Dataset(**dataset_params['val'])
    
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0
    )
    

    
    for idx, sample in enumerate(train_dataloader):
        
        imgs = sample['imgs']
        intrinsics = sample['intrinsics']
        extrinsics = sample['extrinsics']
        sparse_depths = sample['sparse_depths']
        pseudo_depths = sample['pseudo_depths']
        nn_matrix = sample['nn_matrix']
        
        # print(imgs.shape)        #[B, 2, 3, 224, 832]
        # print(intrinsics.shape)  #[B, 2, 3, 3]
        # print(extrinsics.shape)  #[B,2,4,4]
        # print(sparse_depths.shape) #[B,2,H,W]
        # print(pseudo_depths.shape) #[B,2,H,W]
        # print(nn_matrix.shape)     #[B,2,H,W]
        
        warped_left =warp_image2_to_image1(image2=imgs[:,1,:,:,:],depth_map1=pseudo_depths[:,0:1,:,:],
                              K=intrinsics,T=extrinsics)
        
        warped_right = warp_image1_to_image2(image1=imgs[:,0,:,:,:],depth_map2=pseudo_depths[:,1:2,:,:],
                              K=intrinsics,T=extrinsics)
        

        
        plt.subplot(2,2,1)
        plt.axis('off')
        plt.title("left image")
        plt.imshow(imgs[0,0].permute(1,2,0).cpu().numpy())
        plt.subplot(2,2,2)
        plt.axis('off')
        plt.title("right image")
        plt.imshow(imgs[0,1].permute(1,2,0).cpu().numpy())
        
        plt.subplot(2,2,3)
        plt.axis('off')
        plt.title("warped_left")
        plt.imshow(warped_left[0].permute(1,2,0).cpu().numpy())
        plt.subplot(2,2,4)
        plt.axis('off')
        plt.title("right image")
        plt.imshow(warped_right[0].permute(1,2,0).cpu().numpy())    
        
        plt.savefig("1.png")
        
        
        

        
        quit()
    

    