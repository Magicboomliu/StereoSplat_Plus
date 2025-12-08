import torch
import os
import numpy as np
import sys
# Add the codes directory to the path so we can import data module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

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
torch.autograd.set_detect_anomaly(True)
import numpy as np
from torch import Tensor,nn
from mmengine.registry import MODELS
import json
import os
from scipy.spatial.transform import Rotation as R

from tqdm import tqdm
import skimage


def saved_into_json(data_dict,path):
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=4)
        

def interleave_left_right(x: torch.Tensor) -> torch.Tensor:

    first_left_right = x[:, -2:, :, :,:]
    
    rest_views = x[:, :-2, :, :,:]

    B, twoN, C, H, W = rest_views.shape
    assert twoN % 2 == 0, "2N 必须是偶数"
    N = twoN // 2

    left  = rest_views[:, :N]   # (B, N, 3, H, W)
    right = rest_views[:, N:]   # (B, N, 3, H, W)


    # 堆叠后交替
    y = torch.empty_like(rest_views)
    y[:, 0::2] = left
    y[:, 1::2] = right
    
    return torch.cat((y,first_left_right),dim=1)

def interleave_left_right_depth(x: torch.Tensor) -> torch.Tensor:
    
    first_left_right = x[:, -2:, :, :]
    
    rest_views = x[:, :-2, :, :]

    B, twoN, H, W = rest_views.shape
    assert twoN % 2 == 0, "2N 必须是偶数"
    N = twoN // 2

    left  = rest_views[:, :N]   # (B, N, H, W)
    right = rest_views[:, N:]   # (B, N, H, W)

    y = torch.empty_like(rest_views)
    y[:, 0::2] = left
    y[:, 1::2] = right
    return torch.cat((y,first_left_right),dim=1)


def interleave_left_right_pose(x: torch.Tensor) -> torch.Tensor:
    
    first_left_right = x[:, -2:, :, :]
    rest_views = x[:, :-2, :, :]

    B, twoN, H, W = rest_views.shape
    assert twoN % 2 == 0, "2N 必须是偶数"
    assert H == 4 and W == 4, "应该是 4x4 相机矩阵"
    N = twoN // 2

    left  = rest_views[:, :N]   # (B, N, 4, 4)
    right = rest_views[:, N:]   # (B, N, 4, 4)

    y = torch.empty_like(rest_views)
    y[:, 0::2] = left
    y[:, 1::2] = right
    return torch.cat((y,first_left_right),dim=1)


def compute_rotation_angle(c2w_first, c2w_last):
    """
    计算两个c2w矩阵之间的旋转角（欧拉角 x, y, z）
    
    Args:
        c2w_first: 第一个c2w矩阵 (4x4)
        c2w_last: 第二个c2w矩阵 (4x4)
    
    Returns:
        rotation_angles: (x, y, z) 欧拉角（弧度）
    """
    # 提取3x3旋转矩阵（左上角）
    R_first = c2w_first[:3, :3].detach().cpu().numpy()
    R_last = c2w_last[:3, :3].detach().cpu().numpy()
    
    # 计算相对旋转矩阵: R_relative = R_first^-1 * R_last
    R_first_inv = R_first.T  # 对于旋转矩阵，转置等于逆
    R_relative = R_first_inv @ R_last
    
    # 转换为欧拉角 (xyz顺序，单位：弧度)
    rotation = R.from_matrix(R_relative)
    euler_angles = rotation.as_euler('xyz', degrees=False)
    
    return euler_angles  # (x, y, z)


def write_lines_into_txt(lines,path):
    with open(path, 'w') as f:
        for idx, line in enumerate(lines):
            if idx!=len(lines)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
    print(f"Wrote {len(lines)} lines into {path}")




def main(args):
    
    cfg = Config.fromfile(args.config_path)
    cfg.work_dir = args.output_folder
    os.makedirs(args.output_folder, exist_ok=True)
    dataset_config = cfg.dataset_params
    
    
    output_revision_filelist = args.revision_validation_filelist
    os.makedirs(os.path.dirname(output_revision_filelist), exist_ok=True)
    
    
    angle_threshold = args.angle_threshold
    
    
    if cfg.world_center is not None:
        if cfg.world_center=="Center_LiDAR":
            import data.KITTI360_CenterCam_Ref.dataloader as datasets
        elif cfg.world_center=="First_Cam0":
            import data.KITTI360_FirstCam_Ref.dataloader as datasets
        elif cfg.world_center=="First_LiDAR":
            import data.KITTI360_FirstLiDAR_Ref.dataloader as datasets
        elif cfg.world_center=="First_LiDAR_3_Uniform":
            import data.KITTI360_FisrtLiDAR_Random.dataloader as datasets
    
    else:
        import data.KITTI360_CenterCam_Ref.dataloader as datasets
    
    dataset = getattr(datasets, dataset_config.dataset_name)
    
    if args.output_vis:
        val_filelist = args.demo_filelist
    else:
        val_filelist = args.val_filelist
    
    val_params = {
        "datapath":dataset_config.datapath,
        "train_filelist":dataset_config.train_filelist,
        "val_filelist":val_filelist,
        "test_filelist":val_filelist,
        "data_version":dataset_config.data_version,
        "resolution":dataset_config.resolution, 
        "split":"val",
        "sequence":dataset_config.sequence,
        "use_center":dataset_config.use_center,
        "use_first": dataset_config.use_first,
        "use_last": dataset_config.use_last,
        "supp_view_nums": "all",
        "depth_info_dict":dataset_config.depth_info_params,
        "camera_model": dataset_config.camera_model
    }

    
    val_dataset = dataset(**val_params)
    val_dataloader = DataLoader(
        val_dataset, dataset_config.batch_size_val, shuffle=False,
        num_workers=dataset_config.num_workers_val
    )
    
    valid_nums = 0
    
    all_valid_bin_tokens_list = []
    
    for batch_data in tqdm(val_dataloader):
        
        current_bin_token = batch_data['bin_token'][0]
        current_outputs = batch_data['outputs']
        
        current_output_rgbs = current_outputs['rgb']
        current_output_rgbs = interleave_left_right(current_output_rgbs)
        
        current_c2w = current_outputs['c2w']
        current_c2w = interleave_left_right_pose(current_c2w)
        
        
        first_left_c2w = current_c2w[:,-2:,:,:][0][0] # 4x4
        first_left_rgb = current_output_rgbs[:,-2:,:,:,:][0][0].permute(1,2,0).cpu().numpy() # 3x224x832
        
        last_left_c2w = current_c2w[:,-4:-2,:,:][0][0] #4x4
        last_left_rgb = current_output_rgbs[:,-4:-2,:,:,:][0][0].permute(1,2,0).cpu().numpy() # 3x224x832
        
        center_left_c2w = current_c2w[:,-6:-4,:,:][0][0] #4x4
        center_left_rgb = current_output_rgbs[:,-6:-4,:,:,:][0][0].permute(1,2,0).cpu().numpy() # 3x224x832
        
        
        first_center_last_concated_by_height = np.concatenate([first_left_rgb, center_left_rgb, last_left_rgb], axis=0)
        first_center_last_concated_by_height_uint8 = (first_center_last_concated_by_height  * 255.0).astype(np.uint8)

        

        # 计算 last_left_c2w 和 first_left_c2w 之间的旋转角
        rotation_angles = compute_rotation_angle(first_left_c2w, last_left_c2w)
        rot_x, rot_y, rot_z = rotation_angles
        
        # 转换为度数并规范化到 0-180 度范围
        deg_x = np.degrees(rot_x)
        deg_y = np.degrees(rot_y)
        deg_z = np.degrees(rot_z)
        
        # 规范化到 0-180 度范围
        # 先将角度规范化到 0-360，然后如果 > 180，转换为 180-360 的补角
        deg_x_360 = (deg_x % 360 + 360) % 360
        deg_y_360 = (deg_y % 360 + 360) % 360
        deg_z_360 = (deg_z % 360 + 360) % 360
        
        # 转换为 0-180 度范围
        deg_x_180 = deg_x_360 if deg_x_360 <= 180 else 360 - deg_x_360
        deg_y_180 = deg_y_360 if deg_y_360 <= 180 else 360 - deg_y_360
        deg_z_180 = deg_z_360 if deg_z_360 <= 180 else 360 - deg_z_360
        
        
        if np.abs(deg_x_180) < angle_threshold and np.abs(deg_y_180) < angle_threshold:
            all_valid_bin_tokens_list.append(current_bin_token)            
            valid_nums += 1
        else:
            # visuaze
            pass
            #skimage.io.imsave(os.path.join(args.output_folder, f"{current_bin_token}.png"), first_center_last_concated_by_height_uint8)
            

    print(valid_nums)
    print(len(val_dataloader))           


    write_lines_into_txt(all_valid_bin_tokens_list, output_revision_filelist)
        


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path')
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--val_filelist', type=str, default='')
    parser.add_argument('--demo_filelist', type=str, default='')
    
    parser.add_argument('--ablation_type', type=str)
    parser.add_argument('--dataset_type', type=str)
    
    parser.add_argument('--revision_validation_filelist', type=str, default='')
    
    parser.add_argument('--angle_threshold', type=int, default=15)
    

    parser.add_argument(
        "--output_vis",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    ) # visualize the outputs image
    
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    main(args)