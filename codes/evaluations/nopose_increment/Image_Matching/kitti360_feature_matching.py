import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from diffusers.optimization import get_scheduler
import math
import mmcv
import mmengine
from mmengine import MMLogger
from mmengine.config import Config
import logging
from tqdm import tqdm
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import set_seed, convert_outputs_to_fp32, DistributedType, ProjectConfiguration, InitProcessGroupKwargs
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("..")
torch.autograd.set_detect_anomaly(True)
import numpy as np
from torch import Tensor,nn
from tools.metrics import RGB_Quality_Meter,Depth_Quality_Meter,saved_into_json
from mmengine.registry import MODELS
import json
# define the models
from models_lab.VolumeFusion.volumefusion import VolumeFusion
import os

# 内存优化配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 帮助调试内存问题

# 设置PyTorch内存分配策略
torch.backends.cudnn.benchmark = False  # 减少内存碎片
torch.backends.cudnn.deterministic = True

# 启用梯度检查点以节省内存
torch.utils.checkpoint.checkpoint_impl = "reentrant"
from evaluations.data_container.simple_datareader import get_inputs_info,Get_First_Key_Frame_LiDAR_To_World
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import cv2
from model.utils.image import resize_image,HWC3
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import skimage.io
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForKeypointMatching
from pose_estimator import estimate_pose_depth2depth

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

class DepthInfoParams(object):
    def __init__(self,use_pseudo_depth,
                 pseudo_depth_type,
                 use_sparse_lidar):
        self.use_pseudo_depth = use_pseudo_depth
        self.pseudo_depth_type = pseudo_depth_type
        self.use_sparse_lidar  = use_sparse_lidar


if __name__=="__main__":
    root_datapath = "/data1/StereoDatasets/KITTI/KITTI360"
    
    example_pkl_path = "/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/semi_global_maps/2013_05_28_drive_0000_sync/semi_global_0.pkl"
    semi_global_info = load_pkl(example_pkl_path)
    key_input_frames_idx, all_frames_idx = semi_global_info['key_frames_list'], semi_global_info['all_frames_list']

    # LiDAR to World(This World is the True World Coordinate)
    first_key_frame_lidar_to_world_pose = Get_First_Key_Frame_LiDAR_To_World(root_datapath,
                                                                                 key_input_frames_idx[0].replace("annotations","annotations_simple"))

    depth_info_params = DepthInfoParams(
        use_pseudo_depth=True,
        pseudo_depth_type='NMRFStereo',
        use_sparse_lidar=True
    )
    
    # Fixing the Pose Problem using 2D-3D PnP 
    input_cams_data_list = []
    input_cams_to_world_list = []
    input_psudo_depth_list = []
    input_cks_list = []
    
    input_frame_index = 0
    
    for input_frame_name in tqdm(key_input_frames_idx):    
        input_annotation_name = input_frame_name.replace("annotations","annotations_simple")
        
        input_infos_list = get_inputs_info(datapath=root_datapath,
                    reso = [224,1088],
                    first_ref=first_key_frame_lidar_to_world_pose,
                    simple_annotation_path_list=[input_annotation_name],
                    depth_info_params =depth_info_params ,
                    extra_list=[])
        
        networks_input_info, networks_out_aux_info = input_infos_list[0]['input'], input_infos_list[0]['output']
        # fixed 
        input_cam2lidar = networks_input_info['c2w'] #(V,4,4)
        input_lidar2world = networks_input_info['lidar_to_world'] #(V,4,4)
        input_cam2world = networks_out_aux_info['c2w'] #(V,4,4)
        input_image_data_tensor = networks_input_info['imgs'] #(V,3,H,W)
        input_cam_instrinsic = networks_input_info['cks'] #(V,3,3)
        input_psuedo_depth_data = networks_input_info['psuedo_depth'] #(V,H,W)
        
        input_cams_data_list.append(input_image_data_tensor)
        input_cams_to_world_list.append(input_cam2world)
        input_psudo_depth_list.append(input_psuedo_depth_data)
        input_cks_list.append(input_cam_instrinsic)
        
        
        input_frame_index = input_frame_index + 1
        
        if input_frame_index>2:
            break


    first_frame_cam_left = input_cams_data_list[0][0].permute(1,2,0).cpu().numpy()
    first_frame_cam_left_np8 = (first_frame_cam_left*255).astype(np.uint8)
    first_frame_cam_left_pil = Image.fromarray(first_frame_cam_left_np8)
    
    
    second_frame_cam_left = input_cams_data_list[1][0].permute(1,2,0).cpu().numpy()
    second_frame_cam_left_np8 = (second_frame_cam_left*255).astype(np.uint8)
    second_frame_cam_left_pil = Image.fromarray(second_frame_cam_left_np8)
    
    first_frame_depth_map = input_psudo_depth_list[0][0]
    second_frame_depth_map = input_psudo_depth_list[1][0]
    
    
    first_frame_cam2world = input_cams_to_world_list[0][0]
    second_frame_cam2world = input_cams_to_world_list[1][0]
    second_frame_to_first_frame_pose = torch.linalg.inv(first_frame_cam2world) @ second_frame_cam2world #(4,4)
    
      
    first_frame_cks = input_cks_list[0][0]

    # proocessing the image using EfficientLoFTR
    processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
    model = AutoModelForKeypointMatching.from_pretrained("zju-community/efficientloftr")


    with torch.no_grad():
        images = [first_frame_cam_left_pil, second_frame_cam_left_pil]

        inputs = processor(images, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process to get keypoints and matches
        image_sizes = [[(image.height, image.width) for image in images]]
        processed_outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.7)
        
        keypoints_1 = processed_outputs[0]['keypoints0'] #(N,2)
        keypoints_2 = processed_outputs[0]['keypoints1'] #(N,2)
        matches = processed_outputs[0]['matching_scores'] #(N,)
        
        processed_outputs[0]['keypoints0'] = keypoints_1
        processed_outputs[0]['keypoints1'] = keypoints_2
        processed_outputs[0]['matching_scores'] = matches

        
        visualized_images = processor.visualize_keypoint_matching(images, processed_outputs)
        
        visualized_images[0].save("visualized_image1.png")
        quit()



        print("begin Pose estimation")    
        estimated_pose = estimate_pose_depth2depth(
            K1=first_frame_cks,
            K2=first_frame_cks,
            depth1=first_frame_depth_map,
            depth2=second_frame_depth_map,
            kps1=keypoints_1,
            kps2=keypoints_2,
            max_depth=25.0,
            huber_delta=0.03,
            max_samples=100000,
            z_match_base=0.05,
            z_match_rel=0.02,
            trim_ratio=0.8,
            normal_facing_thresh=0.1,
            damping=1e-6
        )
        
        print(estimated_pose)
        print("--------------------------------")
        print(second_frame_to_first_frame_pose)
        
        print("Finish Pose estimation")  
        quit()