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
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import open3d as o3d
import matplotlib.pyplot as plt
from pose_estimator import align_one_frame_sim3_icp_from_tensors
from utils import select_confident_points,resize_the_sparse_lidar_torch

def select_confident_points(points: torch.Tensor, conf: torch.Tensor, conf_thresh: float = 0.5):
    """
    Args:
        points: (1, H, W, 3) 点云坐标
        conf:   (1, H, W) 置信度
        conf_thresh: float，阈值，保留大于该值的点

    Returns:
        selected_points: (N, 3)
    """
    # 变成 (H, W)
    conf = conf.squeeze(0)          # (H, W)
    points = points.squeeze(0)      # (H, W, 3)
    # 置信度mask
    mask = conf > conf_thresh       # (H, W)
    # 选点
    selected_points = points[mask]  # (N, 3)

    return selected_points

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def preprocess_image_tensors_for_vggt(image_tensors, target_size=518, mode="crop"):
    """
    Preprocess image tensors using the same logic as load_and_preprocess_images function.
    This function ensures all images have the same shape and are compatible with VGGT model.
    
    Args:
        image_tensors: List of image tensors with shape (V, 3, H, W)
        target_size: Target size for the output images (default: 518)
        mode: Preprocessing mode, either "crop" or "pad"
        
    Returns:
        torch.Tensor: Preprocessed image tensors with shape (N, V, 3, H, W)
    """
    if len(image_tensors) == 0:
        raise ValueError("At least 1 image tensor is required")
    
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")
    
    processed_images = []
    shapes = set()
    
    for img_tensor in image_tensors:
        # img_tensor shape: (V, 3, H, W)
        V, C, H, W = img_tensor.shape
        
        # Process each view in the tensor
        processed_views = []
        for v in range(V):
            view_img = img_tensor[v]  # (3, H, W)
            
            # Convert tensor to PIL Image for processing
            # Convert from (0,1) range to (0,255) range
            view_img_255 = (view_img.permute(1, 2, 0) * 255).clamp(0, 255).byte().cpu().numpy()
            pil_img = Image.fromarray(view_img_255)
            
            # Apply the same preprocessing logic as load_and_preprocess_images
            width, height = pil_img.size
            
            if mode == "pad":
                # Make the largest dimension target_size while maintaining aspect ratio
                if width >= height:
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
                else:
                    new_height = target_size
                    new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
            else:  # mode == "crop"
                # Set width to target_size
                new_width = target_size
                # Calculate height maintaining aspect ratio, divisible by 14
                new_height = round(height * (new_width / width) / 14) * 14
            
            # Resize with new dimensions
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            
            # Convert back to tensor (0, 1 range)
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            view_tensor = to_tensor(pil_img)  # (3, H, W)
            
            # Center crop height if it's larger than target_size (only in crop mode)
            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                view_tensor = view_tensor[:, start_y:start_y + target_size, :]
            
            # For pad mode, pad to make a square of target_size x target_size
            if mode == "pad":
                h_padding = target_size - view_tensor.shape[1]
                w_padding = target_size - view_tensor.shape[2]
                
                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left
                    
                    # Pad with white (value=1.0)
                    view_tensor = torch.nn.functional.pad(
                        view_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )
            
            processed_views.append(view_tensor)
        
        # Stack processed views back into a tensor
        processed_tensor = torch.stack(processed_views, dim=0)  # (V, 3, H, W)
        processed_images.append(processed_tensor)
        shapes.add((processed_tensor.shape[2], processed_tensor.shape[3]))  # (H, W)
    
    # Check if we have different shapes and pad if necessary
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        
        # Pad images if necessary
        padded_images = []
        for img in processed_images:
            V, C, H, W = img.shape
            h_padding = max_height - H
            w_padding = max_width - W
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        processed_images = padded_images
    
    # Stack all processed images
    processed_images = torch.stack(processed_images, dim=0)  # (N, V, 3, H, W)
    
    # Ensure correct shape when single image
    if len(image_tensors) == 1:
        if processed_images.dim() == 4:
            processed_images = processed_images.unsqueeze(0)
    
    return processed_images

def preprocess_depth_tensors_for_vggt(depth_tensors, target_size=518, mode="crop"):
    """
    Preprocess depth tensors using the same logic as load_and_preprocess_images function.
    This function ensures all depth maps have the same shape and are compatible with VGGT model.
    
    Args:
        depth_tensors: List of depth tensors with shape (V, H, W)
        target_size: Target size for the output depth maps (default: 518)
        mode: Preprocessing mode, either "crop" or "pad"
        
    Returns:
        torch.Tensor: Preprocessed depth tensors with shape (N, V, H, W)
    """
    if len(depth_tensors) == 0:
        raise ValueError("At least 1 depth tensor is required")
    
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")
    
    processed_depths = []
    shapes = set()
    
    for depth_tensor in depth_tensors:
        # depth_tensor shape: (V, H, W)
        V, H, W = depth_tensor.shape
        
        # Process each view in the tensor
        processed_views = []
        for v in range(V):
            view_depth = depth_tensor[v]  # (H, W)
            
            # Store original depth range for restoration
            depth_min = view_depth.min()
            depth_max = view_depth.max()
            
            # Normalize depth to 0-255 range for PIL processing
            if depth_max > depth_min:
                view_depth_normalized = ((view_depth - depth_min) / (depth_max - depth_min) * 255).clamp(0, 255).byte()
            else:
                view_depth_normalized = view_depth.byte()
            
            # Convert to PIL Image (grayscale)
            pil_img = Image.fromarray(view_depth_normalized.cpu().numpy(), mode='L')
            
            # Apply the same preprocessing logic as load_and_preprocess_images
            width, height = pil_img.size
            
            if mode == "pad":
                # Make the largest dimension target_size while maintaining aspect ratio
                if width >= height:
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
                else:
                    new_height = target_size
                    new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
            else:  # mode == "crop"
                # Set width to target_size
                new_width = target_size
                # Calculate height maintaining aspect ratio, divisible by 14
                new_height = round(height * (new_width / width) / 14) * 14
            
            # Resize with new dimensions
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            
            # Convert back to tensor (0, 1 range)
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            view_tensor = to_tensor(pil_img)  # (1, H, W)
            
            # Remove channel dimension to get (H, W)
            view_tensor = view_tensor.squeeze(0)
            
            # Restore original depth scale
            if depth_max > depth_min:
                view_tensor = view_tensor * (depth_max - depth_min) + depth_min
            
            # Center crop height if it's larger than target_size (only in crop mode)
            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                view_tensor = view_tensor[start_y:start_y + target_size, :]
            
            # For pad mode, pad to make a square of target_size x target_size
            if mode == "pad":
                h_padding = target_size - view_tensor.shape[0]
                w_padding = target_size - view_tensor.shape[1]
                
                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left
                    
                    # Pad with 0.0 for depth (or could use mean depth)
                    pad_value = 0.0
                    view_tensor = torch.nn.functional.pad(
                        view_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=pad_value
                    )
            
            processed_views.append(view_tensor)
        
        # Stack processed views back into a tensor
        processed_tensor = torch.stack(processed_views, dim=0)  # (V, H, W)
        processed_depths.append(processed_tensor)
        shapes.add((processed_tensor.shape[1], processed_tensor.shape[2]))  # (H, W)
    
    # Check if we have different shapes and pad if necessary
    if len(shapes) > 1:
        print(f"Warning: Found depth maps with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        
        # Pad depth maps if necessary
        padded_depths = []
        for depth in processed_depths:
            V, H, W = depth.shape
            h_padding = max_height - H
            w_padding = max_width - W
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                depth = torch.nn.functional.pad(
                    depth, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
                )
            padded_depths.append(depth)
        processed_depths = padded_depths
    
    # Stack all processed depth maps
    processed_depths = torch.stack(processed_depths, dim=0)  # (N, V, H, W)
    
    # Ensure correct shape when single depth map
    if len(depth_tensors) == 1:
        if processed_depths.dim() == 3:
            processed_depths = processed_depths.unsqueeze(0)
    
    return processed_depths

def update_intrinsics_for_preprocessing(intrinsics, original_resolution, target_size=518, mode="crop"):
    """
    Update intrinsic matrices after image preprocessing (resize and crop).
    This function exactly mirrors the image preprocessing operations.
    
    Args:
        intrinsics: Intrinsic matrices with shape (N, V, 3, 3)
        original_resolution: Original image resolution [H, W]
        target_size: Target size after preprocessing (default: 518)
        mode: Preprocessing mode, either "crop" or "pad"
        
    Returns:
        torch.Tensor: Updated intrinsic matrices with shape (N, V, 3, 3)
    """
    if intrinsics.dim() != 4:
        raise ValueError(f"Expected intrinsics to have 4 dimensions, got {intrinsics.dim()}")
    
    N, V, _, _ = intrinsics.shape
    original_H, original_W = original_resolution
    
    # Step 1: Calculate new dimensions after resize (keeping aspect ratio)
    if mode == "crop":
        # Set width to target_size, calculate height maintaining aspect ratio
        new_width = target_size
        new_height = round(original_H * (new_width / original_W) / 14) * 14  # Make divisible by 14
    else:  # mode == "pad"
        # Make the largest dimension target_size while maintaining aspect ratio
        if original_W >= original_H:
            new_width = target_size
            new_height = round(original_H * (new_width / original_W) / 14) * 14
        else:
            new_height = target_size
            new_width = round(original_W * (new_height / original_H) / 14) * 14
    
    # Step 2: Calculate scale factors for resize
    scale_w = new_width / original_W
    scale_h = new_height / original_H
    
    # Step 3: Calculate crop parameters (if needed)
    if mode == "crop" and new_height > target_size:
        # Center crop: remove excess height
        crop_start_y = (new_height - target_size) // 2
        crop_end_y = crop_start_y + target_size
        final_height = target_size
    else:
        crop_start_y = 0
        crop_end_y = new_height
        final_height = new_height
    
    # Step 4: Update intrinsic matrices
    updated_intrinsics = intrinsics.clone()
    
    for n in range(N):
        for v in range(V):
            # Get current intrinsic matrix
            K = intrinsics[n, v]  # (3, 3)
            
            # Step 4a: Apply resize scaling
            # K_resized = [fx*scale_w, 0, cx*scale_w]
            #            [0, fy*scale_h, cy*scale_h]
            #            [0, 0, 1]
            K_resized = K.clone()
            K_resized[0, 0] *= scale_w  # fx
            K_resized[1, 1] *= scale_h  # fy
            K_resized[0, 2] *= scale_w  # cx
            K_resized[1, 2] *= scale_h  # cy
            
            # Step 4b: Apply crop offset (if cropping was performed)
            if crop_start_y > 0:
                # When cropping, we need to adjust the principal point y
                # The crop removes pixels from the top, so we subtract the offset
                K_resized[1, 2] -= crop_start_y
            
            # Step 4c: Handle final padding if needed (for consistent output size)
            # This would only happen if we need to ensure all images have the same final size
            # For now, we assume the target_size is the final size
            
            updated_intrinsics[n, v] = K_resized
    
    return updated_intrinsics

class DepthInfoParams(object):
    def __init__(self,use_pseudo_depth,
                 pseudo_depth_type,
                 use_sparse_lidar):
        self.use_pseudo_depth = use_pseudo_depth
        self.pseudo_depth_type = pseudo_depth_type
        self.use_sparse_lidar  = use_sparse_lidar

def prepare_supervision_frames(key_input_frames_idx, 
                               all_frames_idx, 
                               current_keyframe_index, 
                               val_params, 
                               first_key_frame_lidar_to_world_pose):
    """
    准备监督帧：获取当前关键帧和下一个关键帧之间的所有帧作为监督数据
    
    Args:
        key_input_frames_idx: 关键帧列表
        all_frames_idx: 所有帧的索引列表
        current_keyframe_index: 当前关键帧在关键帧列表中的索引
        val_params: 验证参数
        first_key_frame_lidar_to_world_pose: 第一个关键帧的LiDAR到世界坐标变换
        
    Returns:
        supervision_frames: 监督帧列表
    """
    supervision_frames = []
    
    # 检查索引范围
    if current_keyframe_index >= len(key_input_frames_idx):
        print(f"Warning: current_keyframe_index {current_keyframe_index} >= len(key_input_frames_idx) {len(key_input_frames_idx)}")
        return supervision_frames
    
    # 获取当前关键帧的路径
    current_keyframe_path = key_input_frames_idx[current_keyframe_index]
    
    # 找到当前关键帧在 all_frames_idx 中的位置
    try:
        current_frame_idx_in_all = all_frames_idx.index(current_keyframe_path)
    except ValueError:
        print(f"Warning: Current keyframe {current_keyframe_path} not found in all_frames_idx")
        return supervision_frames
    
    # 确定监督帧的范围
    if current_keyframe_index + 1 < len(key_input_frames_idx):
        # 还有下一个关键帧
        next_keyframe_path = key_input_frames_idx[current_keyframe_index + 1]
        try:
            next_frame_idx_in_all = all_frames_idx.index(next_keyframe_path)
            # 获取当前关键帧和下一个关键帧之间的所有帧作为监督数据
            supervision_frame_indices = all_frames_idx[current_frame_idx_in_all + 1:next_frame_idx_in_all]
        except ValueError:
            print(f"Warning: Next keyframe {next_keyframe_path} not found in all_frames_idx")
            # 如果找不到下一个关键帧，获取当前关键帧之后的所有帧
            supervision_frame_indices = all_frames_idx[current_frame_idx_in_all + 1:]
    else:
        # 这是最后一个关键帧，获取之后的所有帧
        supervision_frame_indices = all_frames_idx[current_frame_idx_in_all + 1:]
    
    print(f"Current keyframe: {current_keyframe_path}")
    print(f"Supervision frame range: {len(supervision_frame_indices)} frames")
    
    # 处理每个监督帧
    for frame_idx in supervision_frame_indices:
        try:
            # 构建帧的完整路径
            frame_annotation_name = frame_idx.replace("annotations", "annotations_simple")
            
            # 获取帧信息
            frame_infos = get_inputs_info(
                datapath=val_params['datapath'],
                reso=val_params['resolution'],
                first_ref=first_key_frame_lidar_to_world_pose,
                simple_annotation_path_list=[frame_annotation_name],
                depth_info_params=val_params['depth_info_dict'],
                extra_list=[]
            )
            
            if frame_infos:
                supervision_frames.append(frame_infos[0])
                
        except Exception as e:
            print(f"Warning: Failed to load supervision frame {frame_idx}: {e}")
            continue
    
    print(f"Prepared {len(supervision_frames)} supervision frames")
    return supervision_frames

def world2cam_to_cam2world(world2cam: torch.Tensor) -> torch.Tensor:
    """
    Convert world2cam extrinsics to cam2world extrinsics (homogeneous 4x4).

    Args:
        world2cam (torch.Tensor): shape (1, V, 3, 4)

    Returns:
        cam2world (torch.Tensor): shape (1, V, 4, 4)
    """
    assert world2cam.ndim == 4 and world2cam.shape[0] == 1 and world2cam.shape[2:] == (3, 4), \
        f"Expected input shape (1, V, 3, 4), but got {world2cam.shape}"

    R = world2cam[..., :3, :3]   # (1, V, 3, 3)
    t = world2cam[..., :3, 3:]   # (1, V, 3, 1)

    R_inv = R.transpose(-1, -2)  # (1, V, 3, 3)
    t_inv = -R_inv @ t           # (1, V, 3, 1)

    # 上半部分拼接 [R^T | -R^T t]
    top = torch.cat([R_inv, t_inv], dim=-1)  # (1, V, 3, 4)

    # 下半部分 [0,0,0,1]
    bottom = torch.zeros_like(top[..., :1, :])  # (1, V, 1, 4)
    bottom[..., 0, 3] = 1.0

    cam2world = torch.cat([top, bottom], dim=-2)  # (1, V, 4, 4)
    return cam2world

def normalize_robust_torch_bv(conf: torch.Tensor, lo: float = 1.0, hi: float = 99.0, eps: float = 1e-8):
    """
    Robust normalization to [0,1] for confidence maps with shape (B, V, H, W).
    使用分位数裁剪(lo/hi百分位) + min-max 归一化；按每个(B,V)样本独立处理。

    Args:
        conf: torch.Tensor, shape (B, V, H, W)
        lo: 下分位数百分比（默认 1.0）
        hi: 上分位数百分比（默认 99.0）
        eps: 防止除零

    Returns:
        norm: torch.Tensor, shape (B, V, H, W), 数值∈[0,1]
    """
    assert conf.ndim == 4, "conf must be (B, V, H, W)"
    assert 0.0 <= lo < hi <= 100.0, "Require 0 <= lo < hi <= 100"

    B, V, H, W = conf.shape
    device = conf.device
    dtype_in = conf.dtype

    # 为了数值稳定，用 float32 做统计，再转换回原dtype
    x = conf.reshape(B * V, -1).to(torch.float32)

    # 每个样本(行)的 lo/hi 分位数
    q_lo = torch.quantile(x, lo / 100.0, dim=1, keepdim=True)
    q_hi = torch.quantile(x, hi / 100.0, dim=1, keepdim=True)
    # 分位数裁剪（注意 clamp 不能用张量做min/max，这里用 min/max 组合）
    x_clip = torch.minimum(torch.maximum(x, q_lo), q_hi)

    # 每个样本的 min/max
    x_min = x_clip.min(dim=1, keepdim=True).values
    x_max = x_clip.max(dim=1, keepdim=True).values
    # min-max 到 [0,1]
    x_norm = (x_clip - x_min) / (x_max - x_min + eps)
    # 还原形状 & dtype
    out = x_norm.reshape(B, V, H, W).to(dtype_in).to(device)
    return out


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

    val_params = {
        'datapath': root_datapath,
        'resolution': [224,1088],
        'depth_info_dict': depth_info_params
    }

    
    # Loading VGGT Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    vggt_model.eval()
    
    print("VGGT Model Loaded..............")
    

    # Fixing the Pose Problem using 2D-3D PnP 
    input_cams_to_world_list = []
    input_psudo_depth_list = []
    input_sparse_lidar_depth_list = []
    input_cks_list = []
    input_images_list = []
    input_lidar_to_world_list = []
   
    
    input_cams_to_world_to_first_frame_list = []
    
    input_frame_index = 0
    
    
    key_frame_0_annotation_name = key_input_frames_idx[0].replace("annotations","annotations_simple")
    key_frame_1_annotation_name = key_input_frames_idx[1].replace("annotations","annotations_simple")
    


    key_frame_0_infos = get_inputs_info(datapath=root_datapath,
                reso = [224,1088],
                first_ref=first_key_frame_lidar_to_world_pose,
                simple_annotation_path_list=[key_frame_0_annotation_name],
                depth_info_params =depth_info_params ,
                extra_list=[])
    
    key_frame_1_infos = get_inputs_info(datapath=root_datapath,
                reso = [224,1088],
                first_ref=first_key_frame_lidar_to_world_pose,
                simple_annotation_path_list=[key_frame_1_annotation_name],
                depth_info_params =depth_info_params ,
                extra_list=[])


    supervision_frame_infos = prepare_supervision_frames(
                    key_input_frames_idx, all_frames_idx, 
                    input_frame_index, val_params, 
                    first_key_frame_lidar_to_world_pose)
    
    
    # For VGGT Estimation
    all_infos = key_frame_0_infos +  supervision_frame_infos + key_frame_1_infos
    first_frame_cam2world = all_infos[0]['output']['c2w'][0]

    
    extrinsic_gt_list = [(torch.linalg.inv(first_frame_cam2world)@all_infos[idx]['output']['c2w']).unsqueeze(0) for idx in range(len(all_infos))]
    extrinsic_gt = torch.cat(extrinsic_gt_list,dim=0).unsqueeze(0)    
    extrinsic_gt = extrinsic_gt[:,:,0,:,:]
    

    
    for info in all_infos:        
        
        input_image_data_tensor = info['input']['imgs'] #(V,3,H,W)
        input_cam_instrinsic = info['input']['cks'] #(V,3,3)
        input_psuedo_depth_data = info['input']['psuedo_depth'] #(V,H,W)
        input_cam2world = info['output']['c2w'] #(V,4,4)
        input_lidar2world = info['input']['lidar_to_world'] #(V,4,4)
        input_sparse_lidar_gt_tensor = info['input']['sparse_gts']
        
        input_cams_to_world_list.append(input_cam2world)
        input_psudo_depth_list.append(input_psuedo_depth_data)
        input_cks_list.append(input_cam_instrinsic)
        input_images_list.append(input_image_data_tensor)
        input_lidar_to_world_list.append(input_lidar2world)
        
        input_sparse_lidar_depth_list.append(input_sparse_lidar_gt_tensor)

    # get the all relative pose from the current frame to the first frame
    frist_frame_cam2world = input_cams_to_world_list[0]    
    for cam2world in input_cams_to_world_list:
        cam2_firstcam = torch.linalg.inv(frist_frame_cam2world) @ cam2world
        input_cams_to_world_to_first_frame_list.append(cam2_firstcam)
    
    input_images_all_tensors = torch.stack(input_images_list,dim=0) #(16,2,3,H,W)
    
    
    # Preprocess all input images using the same logic as VGGT's load_and_preprocess_images
    preprocessed_images = preprocess_image_tensors_for_vggt(
        input_images_list, 
        target_size=518, 
        mode="crop")

    # Preprocess GT depth data with the same logic to maintain consistency
    preprocessed_depth = preprocess_depth_tensors_for_vggt(
        input_psudo_depth_list, 
        target_size=518, 
        mode="crop")
    
    input_cks_tensors = torch.stack(input_cks_list,dim=0) #（16,2,3,3）
    # Update intrinsic matrices to match the preprocessed images
    original_resolution = val_params['resolution']  # [224, 1088]
    updated_intrinsics = update_intrinsics_for_preprocessing(
        input_cks_tensors, 
        original_resolution, 
        target_size=518, 
        mode="crop"
    )


    vggt_height = preprocessed_depth[0][0].shape[0]
    vggt_width = preprocessed_depth[0][0].shape[1]
    
    # processed the sparse lidar depth path
    resized_gt_lidar_list = []
    for idx in range(len(input_sparse_lidar_depth_list)):
    
        resized_lidar_depth = resize_the_sparse_lidar_torch(depthmap=input_sparse_lidar_depth_list[idx][0],
                                  raw_K=input_cks_list[idx][0],
                                  after_K=updated_intrinsics[idx][0],
                                  height=vggt_height,
                                  width=vggt_width,
                                  depth_range=80)
        resized_gt_lidar_list.append(resized_lidar_depth.unsqueeze(0).unsqueeze(0))
    
    resized_gt_lidar_tensor = torch.cat(resized_gt_lidar_list,dim=0)

    

    # Move all preprocessed data to the same device as the model
    preprocessed_images = preprocessed_images.to(device)
    preprocessed_depth = preprocessed_depth.to(device)
    updated_intrinsics = updated_intrinsics.to(device)
    resized_gt_lidar_tensor = resized_gt_lidar_tensor.to(device)

    preprocessed_images = preprocessed_images[:,0,:,:,:] #(1,3,H,W)
    preprocessed_depth = preprocessed_depth[:,0,:,:] #(1,H,W)
    updated_intrinsics = updated_intrinsics[:,0,:,:] #(1,3,3)
    resized_gt_lidar_tensor = resized_gt_lidar_tensor[:,0,:,:]
    
    valid_mask = resized_gt_lidar_tensor>0
    valid_mask = valid_mask.float()
    
    

    # VGGT Estimation Resuts
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = vggt_model(preprocessed_images)
            depth = predictions['depth']  #(1,N,H,W,1)
            depth_conf = predictions['depth_conf'] #(1,N,H,W)
            pose_enc = predictions["pose_enc"] #(1,N，9)
            images = predictions['images'] #（1,16,H，W，3）
            pcd = predictions['world_points'] #（1,16,H，W，3）
            pcd_conf = predictions['world_points_conf'] #(1,16,H,W)
            
            depth_conf = normalize_robust_torch_bv(depth_conf,lo=2,hi=98)
            pcd_conf = normalize_robust_torch_bv(pcd_conf,lo=2,hi=98)
            # normalize_robust_torch(depth_conf)
            
            vggt_extrinsic, vggt_intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # vggt camera to world
    vggt_extrinsic = world2cam_to_cam2world(vggt_extrinsic) #(1,N,4,4)
    
    vggt_conf_threshold = select_confident_points(points=pcd[0][4],
                            conf=pcd_conf[0][4],
                            conf_thresh=0.5)
    
    os.makedirs("vggt_infered",exist_ok=True)
    os.makedirs("GTs",exist_ok=True)
    

    torch.save(depth,"vggt_infered/vggt_depth.pt")
    torch.save(depth_conf,"vggt_infered/vggt_depth_conf.pt")
    torch.save(images,"vggt_infered/vggt_images.pt")
    torch.save(vggt_extrinsic,"vggt_infered/vggt_extrinsic.pt")
    torch.save(pcd,"vggt_infered/vggt_pcd.pt")
    torch.save(pcd_conf,"vggt_infered/vggt_pcd_conf.pt")
    torch.save(vggt_extrinsic,"vggt_infered/vggt_intrinsic.pt")
    
    torch.save(extrinsic_gt,"GTs/gt_extrinsic.pt")
    torch.save(preprocessed_depth,"GTs/preprocessed_depth.pt")
    torch.save(updated_intrinsics,"GTs/gt_intrinsic.pt")
    torch.save(resized_gt_lidar_tensor,"GTs/gt_sparse_lidar_depth.pt")

    quit()
    
    

    
    # print("Begin here")
    # S, S_inv, T_icp, T_total, info = align_one_frame_sim3_icp_from_tensors(
    #     vggt_extrinsic=vggt_extrinsic[0][3],  # VGGT Pose 4x4
    #     image=preprocessed_images[3],          # VGGT Image (3,H,W)
    #     GT_depth=preprocessed_depth[3],        # GT Depth
    #     GT_K=updated_intrinsics[3],            # GT Camera Instrinsic
    #     VGGT_PCD=pcd[0][3],               # [H,W,3]（VGGT相机系）
    #     VGGT_PCD_Conf=pcd_conf[0][3],     # [H,W] VGGT point Cloud Confidence
    #     min_depth=5.0, max_depth=30.0, conf_thresh=0.80,
    #     step=1,
    #     voxel_list=(0.3, 0.15, 0.07), 
    #     dist_ratio=3.0 , 
    #     max_iters=(40,30,20),
    # )

    # extrinsics_gt_cam2world = T_total @ vggt_extrinsic[0][3]

    print(vggt_extrinsic[0][4])
    print("--------------------------")
    # print(extrinsics_gt_cam2world)
    # print("----------------------------")
    print(extrinsic_gt[0][4])
    print("---------------------------")
    quit()
    






    
    
    
    # Perform the VGGT Pose/ Point Cloud/ Depth Estimation using Resized Images
    
    
    
    
    

    
    

    #     networks_input_info, networks_out_aux_info = input_infos_list[0]['input'], input_infos_list[0]['output']
    #     # fixed 
    #     input_cam2lidar = networks_input_info['c2w'] #(V,4,4)
    #     input_lidar2world = networks_input_info['lidar_to_world'] #(V,4,4)
    #     input_cam2world = networks_out_aux_info['c2w'] #(V,4,4)
    #     input_image_data_tensor = networks_input_info['imgs'] #(V,3,H,W)
    #     input_cam_instrinsic = networks_input_info['cks'] #(V,3,3)
    #     input_psuedo_depth_data = networks_input_info['psuedo_depth'] #(V,H,W)
        
    #     input_cams_data_list.append(input_image_data_tensor)
    #     input_cams_to_world_list.append(input_cam2world)
    #     input_psudo_depth_list.append(input_psuedo_depth_data)
    #     input_cks_list.append(input_cam_instrinsic)
        
    #     input_images_list.append(input_image_data_tensor)
        
        
        
    #     input_frame_index = input_frame_index + 1
        
    #     if input_frame_index>2:
    #         break


    # first_frame_cam_left = input_cams_data_list[0][0].permute(1,2,0).cpu().numpy()
    # first_frame_cam_left_np8 = (first_frame_cam_left*255).astype(np.uint8)
    # first_frame_cam_left_pil = Image.fromarray(first_frame_cam_left_np8)
    
    
    # second_frame_cam_left = input_cams_data_list[1][0].permute(1,2,0).cpu().numpy()
    # second_frame_cam_left_np8 = (second_frame_cam_left*255).astype(np.uint8)
    # second_frame_cam_left_pil = Image.fromarray(second_frame_cam_left_np8)
    
    # first_frame_depth_map = input_psudo_depth_list[0][0]
    # second_frame_depth_map = input_psudo_depth_list[1][0]
    
    
    # first_frame_cam2world = input_cams_to_world_list[0][0]
    # second_frame_cam2world = input_cams_to_world_list[1][0]
    # second_frame_to_first_frame_pose = torch.linalg.inv(first_frame_cam2world) @ second_frame_cam2world #(4,4)
    
    
    # print(len(input_images_list))

    # quit()