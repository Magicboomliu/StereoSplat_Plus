import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import sys
import argparse
import mmcv
import mmengine
import imageio
from mmengine import MMLogger
from mmengine.config import Config
import logging
import sys
sys.path.append("..")
from data.dataloader import nuScenesDataset

from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger('mmengine').setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
import moviepy.editor as mpy
from accelerate import Accelerator
from accelerate.utils import set_seed, convert_outputs_to_fp32, DistributedType, ProjectConfiguration
from tools.visualization import depths_to_colors
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import json

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

def combine_nusc_images(images_tensor):
    """
    Combine a [6,3,H,W] tensor of NuScenes camera views into a 2x3 grid image.
    """
    assert images_tensor.shape[0] == 6, "Expected 6 views in first dimension"

    # Split into two rows
    row1 = torch.cat([images_tensor[2], images_tensor[0], images_tensor[1]], dim=2)  # width-wise
    row2 = torch.cat([images_tensor[5], images_tensor[3], images_tensor[3]], dim=2)

    # Stack rows height-wise
    combined = torch.cat([row1, row2], dim=1)  # now [3, H*2, W*3]

    # Convert to PIL Image
    to_pil = T.ToPILImage()
    return to_pil(combined)

def compute_the_l1_depth(depth_0, depth_1):
    
    return torch.abs(depth_0-depth_1).mean()


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

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name='omni-gs', 
            # config=config,
            init_kwargs={
                "wandb":{'name': cfg.exp_name},
            }
        )

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)


    # build model
    from builder import builder as model_builder
    
    # build the models, analysis the model here
    my_model = model_builder.build(cfg.model).to(accelerator.device)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    dataset_config = cfg.dataset_params

    # generate datasets
    dataset = nuScenesDataset
    demo_dataset = dataset(dataset_config.resolution, split="demo",
                          use_center=dataset_config.use_center,
                          use_first=dataset_config.use_first,
                          use_last=dataset_config.use_last)
    
    
    demo_dataloader = DataLoader(
        demo_dataset, dataset_config.batch_size_test, shuffle=False,
        num_workers=dataset_config.num_workers_test
    ) # 12
    

    # Loaded the model into the accelerate
    my_model, demo_dataloader = accelerator.prepare(
        my_model, demo_dataloader
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
        global_iter = int(path.split("-")[1])
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
        for i_iter, batch in enumerate(demo_dataloader):
            
            # process the current folder
            bin_token_list = batch['bin_token']
            # peform the log inference validations: visualization results inside
            log_val,(rendered_rgb_by_omni_batch,rendered_depth_by_omni_batch), (rendered_rgb_by_volume_batch,rendered_depth_by_volume_batch),(rendered_rgb_by_pixel_batch,rendered_depth_by_pixel_batch),rgb_gt, metric_v2_depth_batch  = my_model.validation_step_with_bin_tokens(batch, args.output_dir,bin_token_list,False)
            
            
            # get the metrics
            
            rendered_rgb_by_omni_list.append(rendered_rgb_by_omni_batch)
            rendered_rgb_by_pixel_list.append(rendered_rgb_by_pixel_batch)
            rendered_rgb_by_volume_list.append(rendered_rgb_by_volume_batch)
            gt_rgb_imgs_list.append(rgb_gt)
            
            rendered_depth_by_omni_list.append(rendered_depth_by_omni_batch)
            rendered_depth_by_pixel_list.append(rendered_depth_by_pixel_batch)
            rendered_depth_by_volume_list.append(rendered_depth_by_volume_batch)
            gt_depth_list.append(metric_v2_depth_batch)

            
            # get the visualization pcd
            
            
            
            

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
    


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--load-from', type=str, default=None)

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    
    main(args=args)
