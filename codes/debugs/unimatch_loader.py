import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from diffusers.optimization import get_scheduler
import math


import json

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
torch.autograd.set_detect_anomaly(True)


import sys
sys.path.append("..")
from depthsplat.src.models.model_warpper import ModelWarpper
from depthsplat.src.models.encoder.unimatch.mv_unimatch import MultiViewUniMatch
import depthsplat.src.datasets_stereo_matching.KITTI360.dataloader as datasets
from safetensors.torch import load_file

from PIL import Image
import PIL


def read_annotation(annotation_filename):

    with open(annotation_filename) as file:
        annotation = json.load(file)

    extrinsic_matrix = torch.as_tensor(annotation["extrinsic_matrix"])
    
    return extrinsic_matrix


def maybe_resize(img, tgt_reso, ck):
    if not isinstance(img, PIL.Image.Image):
        img = Image.fromarray(img)
    resize_flag = False
    if img.height != tgt_reso[0] or img.width != tgt_reso[1]:
        # img.resize((w, h))
        fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
        scale_h, scale_w = tgt_reso[0] / img.height, tgt_reso[1] / img.width
        fx_scaled, fy_scaled, cx_scaled, cy_scaled = fx * scale_w, fy * scale_h, cx * scale_w, cy * scale_h
        ck = np.array([[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]])
        img = img.resize((tgt_reso[1], tgt_reso[0]))
        resize_flag = True
    return np.array(img), ck, resize_flag

if __name__=="__main__":
    
    model_weight_path = "/data1/zliu/feedforward_outputs/DepthSplat/Depth_Estimation_Only/depth_estimation_224x840/checkpoint-90000/model.safetensors"
    
    state_dict = load_file(model_weight_path)  # 返回的是一个 PyTorch state_dict 格式的字典
    

    stripped_state_dict = {
        k.replace("depth_estimator.", "", 1): v for k, v in state_dict.items()
    }
    
    # Model Loading
    depth_estimator_unimatch = MultiViewUniMatch(
            num_scales=1, # default is 1
            upsample_factor=4, # upsample factor is 4
            lowest_feature_resolution=4, # 4
            vit_type="vitb", # 'vits'
            unet_channels=128, # 128
            grid_sample_disable_cudnn=False, # False, Grid Sampling 
        )
    depth_estimator_unimatch.load_state_dict(stripped_state_dict, strict=True)  # 或 strict=False
    print("Loaded the Model Successfully!")
    
    ''' Dataset Loading  '''
    
    left_image = "/data1/StereoDatasets/KITTI/KITTI360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000002501.png"
    right_image = left_image.replace("image_00","image_01")
    
    left_annotations = left_image.replace("data_2d_raw","annotations")
    left_annotations = left_annotations.replace(".png",".json")
    right_annotations = left_annotations.replace("image_00","image_01")
    
    
    
    
    assert os.path.exists(left_image)
    assert os.path.exists(right_image)
    assert os.path.exists(left_annotations)
    assert os.path.exists(right_annotations)