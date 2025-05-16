
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from diffusers.optimization import get_scheduler
import math
import sys
sys.path.append("..")
from data.KITTI360 import dataloader as datasets


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

from tools.visualization import depths_to_colors
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import json
import math
from tqdm import tqdm
import pickle


def compute_depth_mae_mse_sparse(est_depth, gt_depth):
    """
    Compute MAE and MSE between estimated and sparse ground truth depth.

    Args:
        est_depth (Tensor): shape [B, 6, 1, H, W]
        gt_depth (Tensor):  shape [B, 6, 1, H, W], sparse (0 = invalid)

    Returns:
        mae: scalar, mean absolute error over valid pixels
        mse: scalar, mean squared error over valid pixels
    """
    assert est_depth.shape == gt_depth.shape

    valid_mask = gt_depth > 0

    abs_error = torch.abs(est_depth - gt_depth)[valid_mask]
    sq_error = (est_depth - gt_depth).pow(2)[valid_mask]

    if valid_mask.sum() == 0:
        return float('nan'), float('nan')  # no valid pixels

    mae = abs_error.mean().item()
    mse = sq_error.mean().item()

    return mae, mse


def load_pkl_file(path):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def load_the_sparse_gt_lidar(path,scale=256):
    # Read the image in unchanged mode (preserves uint16 format)
    img = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
    img = img/scale
    return img


def compute_depth_errors(gt_depth: np.ndarray, aligned_depth: np.ndarray, valid_mask: np.ndarray):
    """
    Compute MSE and MAE between GT and aligned prediction in valid regions.

    Args:
        gt_depth (np.ndarray): Ground truth depth map, shape [H, W] or [N].
        aligned_depth (np.ndarray): Aligned predicted depth map, same shape as gt_depth.
        valid_mask (np.ndarray): Boolean mask of valid pixels, same shape.

    Returns:
        mse (float): Mean squared error.
        mae (float): Mean absolute error.
    """
    gt_valid = gt_depth[valid_mask]
    pred_valid = aligned_depth[valid_mask]

    mse = np.mean((gt_valid - pred_valid) ** 2)
    mae = np.mean(np.abs(gt_valid - pred_valid))

    return mse, mae


def compute_psnr_ssim(img1, img2):
    """
    img1, img2: tensors of shape [B, 6, 3, H, W], values in [0,1]
    Returns: mean PSNR and SSIM over all views and batch
    """
    B, V, C, H, W = img1.shape
    psnr_vals = []
    ssim_vals = []

    for b in range(B):
        for v in range(V):
            pred = img1[b, v].unsqueeze(0)  # [1, 3, H, W]
            target = img2[b, v].unsqueeze(0)
            psnr_val = psnr(pred, target, data_range=1.0)
            ssim_val = ssim(pred, target, data_range=1.0)
            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)

    return torch.stack(psnr_vals).mean(), torch.stack(ssim_vals).mean()


def saved_into_json(data_dict,path):
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=4)
        
import cv2
def save_depth_batch_as_jet(depth_batch, save_dir="jet_depths"):
    """
    Args:
        depth_batch: torch.Tensor, shape [1, 6, 1, H, W]
        save_dir: directory to save the images
    """
    os.makedirs(save_dir, exist_ok=True)
    depth_batch = depth_batch.squeeze(0).squeeze(2)  # → shape: [6, H, W]
    
    depth_batch = 376/(depth_batch+1e-4)
    depth_batch = torch.clamp(depth_batch,min=0,max=320)
    
    for i, depth in enumerate(depth_batch):
        depth_np = depth.cpu().numpy()  # Convert to NumPy array
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        depth_uint8 = depth_uint8[0]
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        cv2.imwrite(f"{save_dir}/depth_{i:02d}.png", depth_color)

def main(args):
    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, 
        logging_dir=None
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=None,
        project_config=accelerator_project_config,
    )

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)

    #################### Dataset Configurration #############################
    dataset_config = cfg.dataset_params    
    # build model
    from builder import builder as model_builder
    my_model = model_builder.build(cfg.model).to(accelerator.device)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)

    # generate datasets
    dataset = getattr(datasets, dataset_config.dataset_name)

    if args.validation_list!='':
        dataset_config.val_filelist = args.validation_list
    
    val_params = {
        "datapath":dataset_config.datapath,
        "train_filelist":dataset_config.train_filelist,
        "val_filelist":dataset_config.val_filelist,
        "test_filelist":dataset_config.val_filelist,
        "data_version":dataset_config.data_version,
        "resolution":dataset_config.resolution, 
        "split":"val",
        "sequence":dataset_config.sequence,
        "use_center":dataset_config.use_center,
        "use_first": dataset_config.use_first,
        "use_last": dataset_config.use_last,
        
    }
    
    val_dataset = dataset(**val_params)

    val_dataloader = DataLoader(
        val_dataset, dataset_config.batch_size_val, shuffle=False,
        num_workers=dataset_config.num_workers_val
    )
    
    # move to the accelerate
    my_model,val_dataloader= accelerator.prepare(
        my_model, val_dataloader
    )

    # Potentially load in the weights and states from a previous save
    if args.load_from:
        cfg.load_from = args.load_from
    if cfg.load_from:
        path = cfg.load_from
    else:
        path = None

    if path:
        accelerator.print(f"Loading from checkpoint {path}")
        accelerator.load_state(path, map_location='cpu', strict=False)
        global_iter = int((os.path.basename(os.path.normpath(path))).split("-")[1]) 
        print(f'Successfully loaded from iter{global_iter}')
    else:
        print('Can\'t find checkpoint {}. Randomly initialize model parameters anyway.'.format(args.load_from))


    rendered_rgb_by_omni_list = []
    rendered_rgb_by_volume_list = []
    rendered_rgb_by_pixel_list = []
    gt_rgb_imgs_list = []
    
    
    rendered_depth_by_omni_list = []
    rendered_depth_by_volume_list = []
    rendered_depth_by_pixel_list = []
    gt_depth_list = []


    # doing the predicted images 
    with torch.no_grad():
        my_model.eval()
        for batch in tqdm(val_dataloader):
            
            # process the current folder
            bin_token_list = batch['bin_token']
            
            
            # peform the log inference validations: visualization results inside
            log_val,(rendered_rgb_by_omni_batch,rendered_depth_by_omni_batch), (rendered_rgb_by_volume_batch,rendered_depth_by_volume_batch),(rendered_rgb_by_pixel_batch,rendered_depth_by_pixel_batch),rgb_gt, metric_v2_depth_batch  = my_model.validation_step_with_bin_tokens(batch, args.output_dir,bin_token_list,args.output_vis)
            
            # get the metrics
            rendered_rgb_by_omni_list.append(rendered_rgb_by_omni_batch)
            rendered_rgb_by_pixel_list.append(rendered_rgb_by_pixel_batch)
            rendered_rgb_by_volume_list.append(rendered_rgb_by_volume_batch)
            gt_rgb_imgs_list.append(rgb_gt)
            
            rendered_depth_by_omni_list.append(rendered_depth_by_omni_batch)
            rendered_depth_by_pixel_list.append(rendered_depth_by_pixel_batch)
            rendered_depth_by_volume_list.append(rendered_depth_by_volume_batch)
            
            
            projected_depth_tensor_list_batch = []
            for img_name in batch['outputs']['input_image_path']:
                img_name = img_name[0]
                project_lidar_path = img_name.replace("data_2d_raw","projected_sparse_lidar/data_2d_raw")
                
                assert os.path.exists(project_lidar_path)
                project_lidar = load_the_sparse_gt_lidar(project_lidar_path)
                project_lidar = F.interpolate(torch.from_numpy(project_lidar).unsqueeze(0).unsqueeze(0),size=[224,840],
                              mode='bilinear')
                project_lidar = project_lidar.squeeze(0).squeeze(0)
                project_lidar  = project_lidar.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(rendered_depth_by_omni_batch.device)
                projected_depth_tensor_list_batch.append(project_lidar)
                
            
            projected_depth_tensor_batch = torch.cat(projected_depth_tensor_list_batch,dim=1)
            gt_depth_list.append(projected_depth_tensor_batch)
    
    
    
    rendered_rgb_by_omni_all = torch.cat(rendered_rgb_by_omni_list,dim=0).cpu()
    rendered_rgb_by_pixel_all = torch.cat(rendered_rgb_by_pixel_list,dim=0).cpu()
    rendered_rgb_by_volume_all = torch.cat(rendered_rgb_by_volume_list,dim=0).cpu()
    gt_rgb_all = torch.cat(gt_rgb_imgs_list,dim=0).cpu()
    
    rendered_depth_by_omni_all = torch.cat(rendered_depth_by_omni_list,dim=0).cpu()
    rendered_depth_by_pixel_all = torch.cat(rendered_depth_by_pixel_list,dim=0).cpu()
    rendered_depth_by_volume_all = torch.cat(rendered_depth_by_volume_list,dim=0).cpu()
    gt_depth_all = torch.cat(gt_depth_list,dim=0).cpu()
    
    mae_omni, mse_omni = compute_depth_mae_mse_sparse(est_depth=rendered_depth_by_omni_all,gt_depth=gt_depth_all)
    mae_pixel, mse_pixel = compute_depth_mae_mse_sparse(est_depth=rendered_depth_by_pixel_all,gt_depth=gt_depth_all)
    mae_volume, mse_volume = compute_depth_mae_mse_sparse(est_depth=rendered_depth_by_volume_all,gt_depth=gt_depth_all)
    
    ## PSNR and SSIM
    psnr_omni, ssim_omni = compute_psnr_ssim(rendered_rgb_by_omni_all,gt_rgb_all)
    psnr_pixel, ssim_pixel = compute_psnr_ssim(rendered_rgb_by_pixel_all,gt_rgb_all)
    psnr_volume, ssim_volumne = compute_psnr_ssim(rendered_rgb_by_volume_all,gt_rgb_all)
    
    sub_folder_omni = os.path.join(args.output_dir,"omni")
    sub_folder_pixel = os.path.join(args.output_dir,"pixel")
    sub_folder_volume = os.path.join(args.output_dir,"volume")
    
    os.makedirs(sub_folder_omni,exist_ok=True)
    os.makedirs(sub_folder_pixel,exist_ok=True)
    os.makedirs(sub_folder_volume,exist_ok=True)
    
        
    saved_into_json(data_dict={"psnr":psnr_omni.item(),"ssim":ssim_omni.item(),
                               "depth_mae": mae_omni-3                               
                               },
                    path = os.path.join(sub_folder_omni,"metric.json"))    
    
    saved_into_json(data_dict={"psnr":psnr_pixel.item(),"ssim":ssim_pixel.item(),
                               "depth_mae": mae_pixel-3
                               },
                    path = os.path.join(sub_folder_pixel,"metric.json"))
    
    saved_into_json(data_dict={"psnr":psnr_volume.item(),"ssim":ssim_volumne.item(),
                               "depth_mae":mae_volume-3
                               },
                    path = os.path.join(sub_folder_volume,"metric.json"))



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--load_from', type=str, default='')
    parser.add_argument("--validation_list",type=str,default='') # use larger list for evaluation
    parser.add_argument(
        "--output_vis",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    ) # visualize the outputs images

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus

    main(args)
