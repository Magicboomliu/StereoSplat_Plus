import os
import numpy as np
import argparse
import torch 
import sys
import os, time, argparse, os.path as osp, numpy as np
sys.path.append("..")
torch.autograd.set_detect_anomaly(True)
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



    
    

def analyze_batch_data(batch):
    
    
    output_batch_dict = dict()
    
    
    


    bin_token_name = batch['bin_token']
    input_cam_batch_data = batch['inputs_pix']                                 
    input_batch_data = batch['inputs']
    
    print(bin_token_name)
    quit()
    
    input_rgb =  input_batch_data['rgb'] # torch.Size([1, 2, 3, 224, 840]) #(B,V,3,H,W)
    input_camera_intrinsics = input_cam_batch_data['ck'] #(B,V,3,3) 
    input_camera_extrinsics = input_cam_batch_data['c2w'] #(B,V,4,4)
    
    input_psuedo_depth = input_cam_batch_data['depth_m'] #(B,V,H,W)
    input_sparse_depth = input_cam_batch_data['sparse_gt_depth'] #(B,V,H,W)
    
    
    camera_intrinsics_matrix = input_camera_intrinsics[0][0]
    
    output_batch_dict["output_imgs"] = batch["outputs"]["rgb"]
    output_batch_dict["output_depths"] = batch["outputs"]["depth"]
    output_batch_dict["output_depths_m"] = batch["outputs"]["depth_m"]
    output_batch_dict["output_confs_m"] = batch["outputs"]["conf_m"]       
    output_batch_dict["output_positions"] = (batch["outputs"]["rays_o"] + batch["outputs"]["rays_d"] * \
                        batch["outputs"]["depth_m"].unsqueeze(-1))
    output_batch_dict["output_rays_o"] = batch["outputs"]["rays_o"]
    output_batch_dict["output_rays_d"] = batch["outputs"]["rays_d"]
    output_batch_dict["output_c2ws"] = batch["outputs"]["c2w"]
    output_batch_dict["output_fovxs"] = batch["outputs"]["fovx"]
    output_batch_dict["output_fovys"] = batch["outputs"]["fovy"]
    output_batch_dict['output_sparse_depth'] = batch['outputs']['sparse_gt_depth']
    
    
    
    output_images = interleave_left_right(output_batch_dict["output_imgs"])
    output_depths = interleave_left_right_depth(output_batch_dict["output_depths"])

    output_c2ws = interleave_left_right_pose(output_batch_dict["output_c2ws"])
    
    
    torch.save(output_images, "output_images.pth")
    torch.save(output_depths, "output_depths.pth")
    torch.save(output_c2ws, "output_c2ws.pth")
    
    torch.save(input_rgb,"input_rgb.pth")
    
    
    quit()
    
    
    print(output_images.shape) #[1,V,3,H,W]
    
    print(output_depths.shape) # [1,V,H,W]
    print(output_c2ws.shape)   # [1,V,4,4]
    
    quit()
    
    







def main(args):
    
    cfg = Config.fromfile(args.config_path)
    cfg.work_dir = args.output_folder
    

    logger_mm = MMLogger.get_instance('mmengine', log_level='WARNING')
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.work_dir, 
        logging_dir=os.path.join(cfg.work_dir, 'logs')
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)
        
    
    dataset_config = cfg.dataset_params
    # configure logger
    if accelerator.is_main_process:
        os.makedirs(args.output_folder, exist_ok=True)
        cfg.dump(osp.join(args.output_folder, osp.basename(args.config_path)))
        
        
    
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
    
    
    
    val_filelist = args.val_filelist


    all_params = {
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
    
    

    all_dataset = dataset(**all_params)
    all_dataloader = DataLoader(
        all_dataset, dataset_config.batch_size_val, shuffle=False,
        num_workers=dataset_config.num_workers_val
    )


    for batch in tqdm(all_dataloader):
        # process the current folder
        bin_token_list = batch['bin_token']
        
        
        analyze_batch_data(batch)
        



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
    
    
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus

    main(args)