
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
            gt_depth_list.append(metric_v2_depth_batch)
    
    
    rendered_rgb_by_omni_all = torch.cat(rendered_rgb_by_omni_list,dim=0).cpu()
    rendered_rgb_by_pixel_all = torch.cat(rendered_rgb_by_pixel_list,dim=0).cpu()
    rendered_rgb_by_volume_all = torch.cat(rendered_rgb_by_volume_list,dim=0).cpu()
    gt_rgb_all = torch.cat(gt_rgb_imgs_list,dim=0).cpu()
    
    rendered_depth_by_omni_all = torch.cat(rendered_depth_by_omni_list,dim=0).cpu()
    rendered_depth_by_pixel_all = torch.cat(rendered_depth_by_pixel_list,dim=0).cpu()
    rendered_depth_by_volume_all = torch.cat(rendered_depth_by_volume_list,dim=0).cpu()
    gt_depth_all = torch.cat(gt_depth_list,dim=0).cpu()
    
    
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
    
        
    saved_into_json(data_dict={"psnr":psnr_omni.item(),"ssim":ssim_omni.item()},
                    path = os.path.join(sub_folder_omni,"metric.json"))    
    
    saved_into_json(data_dict={"psnr":psnr_pixel.item(),"ssim":ssim_pixel.item()},
                    path = os.path.join(sub_folder_pixel,"metric.json"))
    
    saved_into_json(data_dict={"psnr":psnr_volume.item(),"ssim":ssim_volumne.item()},
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
