
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
import matplotlib.pyplot as plt

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

    B, C, H, W = img1.shape
    psnr_vals = []
    ssim_vals = []

    for b in range(B):
            pred = img1[b].unsqueeze(0)  # [1, 3, H, W]
            target = img2[b].unsqueeze(0)
            psnr_val = psnr(pred, target, data_range=1.0)
            ssim_val = ssim(pred, target, data_range=1.0)
            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)

    return torch.stack(psnr_vals).mean().data.item(), torch.stack(ssim_vals).mean().data.item()


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



def seperate_rendered_views(mode='FLC',batch_results=None):
    # make sure the input tensor shape is [B,V,3,H,W]
    # type select from rgb or depth
    
    results_dict = {}
    
    if mode=='FLC':
        results_dict["center_l"] = batch_results[:,4,:,:,:] # [B,3,H,W]
        results_dict["center_r"] = batch_results[:,5,:,:,:] # [B,3,H,W]
        results_dict["first_l"] = batch_results[:,0,:,:,:] # [B,3,H,W]
        results_dict["first_r"] = batch_results[:,2,:,:,:] # [B,3,H,W]
        
        results_dict["last_l"] = batch_results[:,1,:,:,:] # [B,3,H,W]
        results_dict["last_r"] = batch_results[:,3,:,:,:] # [B,3,H,W]
    
    
    else:
        raise NotImplementedError

    return results_dict


class Basic_Meter(object):
    def __init__(self,psnr,ssim,mae,mse):
        self.psnr = psnr
        self.ssim = ssim
        self.mae = mae
        self.mse = mse
        self.counter = 0
    
    def update(self,psnr,ssim,mae,mse):
        self.psnr +=psnr
        self.ssim +=ssim
        self.mae +=mae
        self.mse +=mse
        self.counter = self.counter+1
    
    def get_stats(self):
        if self.counter ==0:
            return{
            "psnr": 0,
            "ssim": 0,
            "mae": 0,
            "mse":0
                
            }
        else:
            return {
                "psnr": self.psnr/self.counter,
                "ssim": self.ssim/self.counter,
                "mae": self.mae/self.counter,
                "mse":self.mse/self.counter
            }
        
def convert_depth_to_disp(factor=328.318735,depth=None):
    
    mask = depth>0
    mask = mask.astype(np.float32)

    disparity = factor / (depth +1e-3)
    disparity = disparity * mask
    disparity = np.clip(disparity,a_max=220,a_min=0)
    
    disparity = kitti_colormap(disparity)
    return disparity


def kitti_colormap(disparity, maxval=-1):
	"""
	A utility function to reproduce KITTI fake colormap
	Arguments:
	  - disparity: numpy float32 array of dimension HxW
	  - maxval: maximum disparity value for normalization (if equal to -1, the maximum value in disparity will be used)
	
	Returns a numpy uint8 array of shape HxWx3.
	"""
	if maxval < 0:
		maxval = np.max(disparity)

	colormap = np.asarray([[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],[0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0]])
	weights = np.asarray([8.771929824561404,5.405405405405405,8.771929824561404,5.747126436781609,8.771929824561404,5.405405405405405,8.771929824561404,0])
	cumsum = np.asarray([0,0.114,0.299,0.413,0.587,0.701,0.8859999999999999,0.9999999999999999])

	colored_disp = np.zeros([disparity.shape[0], disparity.shape[1], 3])
	values = np.expand_dims(np.minimum(np.maximum(disparity/maxval, 0.), 1.), -1)
	bins = np.repeat(np.repeat(np.expand_dims(np.expand_dims(cumsum,axis=0),axis=0), disparity.shape[1], axis=1), disparity.shape[0], axis=0)
	diffs = np.where((np.repeat(values, 8, axis=-1) - bins) > 0, -1000, (np.repeat(values, 8, axis=-1) - bins))
	index = np.argmax(diffs, axis=-1)-1

	w = 1-(values[:,:,0]-cumsum[index])*np.asarray(weights)[index]


	colored_disp[:,:,2] = (w*colormap[index][:,:,0] + (1.-w)*colormap[index+1][:,:,0])
	colored_disp[:,:,1] = (w*colormap[index][:,:,1] + (1.-w)*colormap[index+1][:,:,1])
	colored_disp[:,:,0] = (w*colormap[index][:,:,2] + (1.-w)*colormap[index+1][:,:,2])

	return (colored_disp*np.expand_dims((disparity>0),-1)*255).astype(np.uint8)


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
        "use_stereo": dataset_config.use_stereo
        
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
        print('Successfully loaded from {}'.format(path))
    else:
        print('Can\'t find checkpoint {}. Randomly initialize model parameters anyway.'.format(args.load_from))
    
    meter_rendered_results_omni_dict = {
        "center_l": Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'center_r': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'first_l': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'first_r': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'last_l': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'last_r': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0)
    }

    meter_rendered_results_pixel_dict = {
        "center_l": Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'center_r': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'first_l': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'first_r': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'last_l': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'last_r': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0)
    }

    meter_rendered_results_volume_dict = {
        "center_l": Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'center_r': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'first_l': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'first_r': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'last_l': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'last_r': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0)
    }

    def update_results_metrics(meter_dict,pred_rgb,gt_rgb,pred_depth,gt_depth):
        for key in meter_dict.keys():
            psnr, ssim = compute_psnr_ssim(img1=pred_rgb[key].cpu(),
                                              img2=gt_rgb[key].cpu())
            mse,mae = compute_depth_errors(gt_depth=gt_depth[key].cpu().numpy(),
                                                 aligned_depth=pred_depth[key].cpu().numpy(),
                                                 valid_mask=(gt_depth[key].cpu().numpy()>0))
            meter_dict[key].update(psnr,ssim,mae,mse)

    def output_meter_into_dict(meter_dict):
        for key in meter_dict.keys():
            meter_dict[key] =  meter_dict[key].get_stats()
        return meter_dict


    sub_folder_omni = os.path.join(args.output_dir,"omni")
    sub_folder_pixel = os.path.join(args.output_dir,"pixel")
    sub_folder_volume = os.path.join(args.output_dir,"volume")
    sub_folder_GT = os.path.join(args.output_dir,"GT")
    
    os.makedirs(sub_folder_omni,exist_ok=True)
    os.makedirs(sub_folder_pixel,exist_ok=True)
    os.makedirs(sub_folder_volume,exist_ok=True)
    os.makedirs(sub_folder_GT,exist_ok=True)

    

    # doing the predicted images 
    with torch.no_grad():
        my_model.eval()
        for batch in tqdm(val_dataloader):
            # process the current folder
            bin_token_list = batch['bin_token']

            # peform the log inference validations: visualization results inside
            log_val,(rendered_rgb_by_omni_batch,rendered_depth_by_omni_batch), \
            (rendered_rgb_by_volume_batch,rendered_depth_by_volume_batch), \
                (rendered_rgb_by_pixel_batch,rendered_depth_by_pixel_batch),\
                    rgb_gt, sparse_depth_batch  = my_model.validation_step_with_bin_tokens(batch, args.output_dir,
                                                                                              bin_token_list,args.output_vis)
            
            if dataset_config.use_center:
                # here the input is the 6 channels,[B,V,3,H,W]
                rgb_sep_rendered_omni_batch = seperate_rendered_views(mode='FLC',
                                                                  batch_results=rendered_rgb_by_omni_batch)
                depth_sep_rendered_omni_batch = seperate_rendered_views(mode='FLC',
                                                                  batch_results=rendered_depth_by_omni_batch)
                
                # ---    
                rgb_sep_rendered_volume_batch = seperate_rendered_views(mode='FLC',
                                                                  batch_results=rendered_rgb_by_volume_batch)
                depth_sep_rendered_volume_batch = seperate_rendered_views(mode='FLC',
                                                                  batch_results=rendered_depth_by_volume_batch)
                # --- 
                rgb_sep_rendered_pixel_batch = seperate_rendered_views(mode='FLC',
                                                                  batch_results=rendered_rgb_by_pixel_batch)
                depth_sep_rendered_pixel_batch = seperate_rendered_views(mode='FLC',
                                                                  batch_results=rendered_depth_by_pixel_batch)
                # --- 
                rgb_sep_rendered_gt_batch = seperate_rendered_views(mode='FLC',
                                                                  batch_results=rgb_gt)
                depth_sep_rendered_gt_batch = seperate_rendered_views(mode='FLC',
                                                                  batch_results=sparse_depth_batch)
                
                
                update_results_metrics(meter_dict=meter_rendered_results_omni_dict,
                                       pred_rgb=rgb_sep_rendered_omni_batch,
                                       pred_depth=depth_sep_rendered_omni_batch,
                                       gt_rgb=rgb_sep_rendered_gt_batch,
                                       gt_depth=depth_sep_rendered_gt_batch)
                
                update_results_metrics(meter_dict=meter_rendered_results_volume_dict,
                                       pred_rgb=rgb_sep_rendered_volume_batch,
                                       pred_depth=depth_sep_rendered_volume_batch,
                                       gt_rgb=rgb_sep_rendered_gt_batch,
                                       gt_depth=depth_sep_rendered_gt_batch)
                
                update_results_metrics(meter_dict=meter_rendered_results_pixel_dict,
                                       pred_rgb=rgb_sep_rendered_pixel_batch,
                                       pred_depth=depth_sep_rendered_pixel_batch,
                                       gt_rgb=rgb_sep_rendered_gt_batch,
                                       gt_depth=depth_sep_rendered_gt_batch)
                
                
                
                if args.output_vis:
                    # Omni-Scene
                    plt.figure(figsize=(20,10))
                    plt.subplot(3,2,1)
                    plt.axis('off')
                    plt.title("F-(l)")
                    plt.imshow(rgb_sep_rendered_omni_batch['first_l'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,2)
                    plt.axis('off')
                    plt.title("F-(r)")
                    plt.imshow(rgb_sep_rendered_omni_batch['first_r'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,3)
                    plt.axis('off')
                    plt.title("C-(l):Input")
                    plt.imshow(rgb_sep_rendered_omni_batch['center_l'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,4)
                    plt.axis('off')
                    plt.title("C-(r):Input")
                    plt.imshow(rgb_sep_rendered_omni_batch['center_r'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,5)
                    plt.axis('off')
                    plt.title("L-(l)")
                    plt.imshow(rgb_sep_rendered_omni_batch['last_l'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,6)
                    plt.axis('off')
                    plt.title("L-(r)")
                    plt.imshow(rgb_sep_rendered_omni_batch['last_r'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.savefig(os.path.join(sub_folder_omni,bin_token_list[0][:-4]+"_omni.png"),bbox_inches='tight')
                    
                    # Omni-Scene Disparity
                    plt.figure(figsize=(20,10))
                    plt.subplot(3,2,1)
                    plt.axis('off')
                    plt.title("F-(l)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_omni_batch['first_l'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,2)
                    plt.axis('off')
                    plt.title("F-(r)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_omni_batch['first_r'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,3)
                    plt.axis('off')
                    plt.title("C-(l):Input")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_omni_batch['center_l'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,4)
                    plt.axis('off')
                    plt.title("C-(r):Input")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_omni_batch['center_r'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,5)
                    plt.axis('off')
                    plt.title("L-(l)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_omni_batch['last_l'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,6)
                    plt.axis('off')
                    plt.title("L-(r)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_omni_batch['last_r'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.savefig(os.path.join(sub_folder_omni,bin_token_list[0][:-4]+"_depth_omni.png"),bbox_inches='tight')

                    
                    # Pixel
                    plt.figure(figsize=(20,10))
                    plt.subplot(3,2,1)
                    plt.axis('off')
                    plt.title("F-(l)")
                    plt.imshow(rgb_sep_rendered_pixel_batch['first_l'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,2)
                    plt.axis('off')
                    plt.title("F-(r)")
                    plt.imshow(rgb_sep_rendered_pixel_batch['first_r'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,3)
                    plt.axis('off')
                    plt.title("C-(l):Input")
                    plt.imshow(rgb_sep_rendered_pixel_batch['center_l'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,4)
                    plt.axis('off')
                    plt.title("C-(r):Input")
                    plt.imshow(rgb_sep_rendered_pixel_batch['center_r'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,5)
                    plt.axis('off')
                    plt.title("L-(l)")
                    plt.imshow(rgb_sep_rendered_pixel_batch['last_l'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,6)
                    plt.axis('off')
                    plt.title("L-(r)")
                    plt.imshow(rgb_sep_rendered_pixel_batch['last_r'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.savefig(os.path.join(sub_folder_pixel,bin_token_list[0][:-4]+"_pixel.png"),bbox_inches='tight')
                    

                    # Pixel Disparity
                    plt.figure(figsize=(20,10))
                    plt.subplot(3,2,1)
                    plt.axis('off')
                    plt.title("F-(l)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_pixel_batch['first_l'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,2)
                    plt.axis('off')
                    plt.title("F-(r)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_pixel_batch['first_r'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,3)
                    plt.axis('off')
                    plt.title("C-(l):Input")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_pixel_batch['center_l'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,4)
                    plt.axis('off')
                    plt.title("C-(r):Input")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_pixel_batch['center_r'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,5)
                    plt.axis('off')
                    plt.title("L-(l)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_pixel_batch['last_l'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,6)
                    plt.axis('off')
                    plt.title("L-(r)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_pixel_batch['last_r'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.savefig(os.path.join(sub_folder_pixel,bin_token_list[0][:-4]+"_depth_pixel.png"),bbox_inches='tight')

                    
                    
                    # Voxel
                    plt.figure(figsize=(20,10))
                    plt.subplot(3,2,1)
                    plt.axis('off')
                    plt.title("F-(l)")
                    plt.imshow(rgb_sep_rendered_volume_batch['first_l'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,2)
                    plt.axis('off')
                    plt.title("F-(r)")
                    plt.imshow(rgb_sep_rendered_volume_batch['first_r'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,3)
                    plt.axis('off')
                    plt.title("C-(l):Input")
                    plt.imshow(rgb_sep_rendered_volume_batch['center_l'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,4)
                    plt.axis('off')
                    plt.title("C-(r):Input")
                    plt.imshow(rgb_sep_rendered_volume_batch['center_r'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,5)
                    plt.axis('off')
                    plt.title("L-(l)")
                    plt.imshow(rgb_sep_rendered_volume_batch['last_l'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,6)
                    plt.axis('off')
                    plt.title("L-(r)")
                    plt.imshow(rgb_sep_rendered_volume_batch['last_r'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.savefig(os.path.join(sub_folder_volume,bin_token_list[0][:-4]+"_voxel.png"),bbox_inches='tight')


                    # Voxel Disparity
                    plt.figure(figsize=(20,10))
                    plt.subplot(3,2,1)
                    plt.axis('off')
                    plt.title("F-(l)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_volume_batch['first_l'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,2)
                    plt.axis('off')
                    plt.title("F-(r)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_volume_batch['first_r'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,3)
                    plt.axis('off')
                    plt.title("C-(l):Input")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_volume_batch['center_l'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,4)
                    plt.axis('off')
                    plt.title("C-(r):Input")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_volume_batch['center_r'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,5)
                    plt.axis('off')
                    plt.title("L-(l)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_volume_batch['last_l'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,6)
                    plt.axis('off')
                    plt.title("L-(r)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_volume_batch['last_r'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.savefig(os.path.join(sub_folder_volume,bin_token_list[0][:-4]+"_depth_volume.png"),bbox_inches='tight')

                    


                    # RGB
                    plt.figure(figsize=(20,10))
                    plt.subplot(3,2,1)
                    plt.axis('off')
                    plt.title("F-(l)")
                    plt.imshow(rgb_sep_rendered_gt_batch['first_l'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,2)
                    plt.axis('off')
                    plt.title("F-(r)")
                    plt.imshow(rgb_sep_rendered_gt_batch['first_r'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,3)
                    plt.axis('off')
                    plt.title("C-(l):Input")
                    plt.imshow(rgb_sep_rendered_gt_batch['center_l'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,4)
                    plt.axis('off')
                    plt.title("C-(r):Input")
                    plt.imshow(rgb_sep_rendered_gt_batch['center_r'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,5)
                    plt.axis('off')
                    plt.title("L-(l)")
                    plt.imshow(rgb_sep_rendered_gt_batch['last_l'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,6)
                    plt.axis('off')
                    plt.title("L-(r)")
                    plt.imshow(rgb_sep_rendered_gt_batch['last_r'].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.savefig(os.path.join(sub_folder_GT,bin_token_list[0][:-4]+"_GT.png"),bbox_inches='tight')
                    
                    # Voxel Disparity
                    plt.figure(figsize=(20,10))
                    plt.subplot(3,2,1)
                    plt.axis('off')
                    plt.title("F-(l)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_gt_batch['first_l'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,2)
                    plt.axis('off')
                    plt.title("F-(r)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_gt_batch['first_r'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,3)
                    plt.axis('off')
                    plt.title("C-(l):Input")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_gt_batch['center_l'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,4)
                    plt.axis('off')
                    plt.title("C-(r):Input")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_gt_batch['center_r'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,5)
                    plt.axis('off')
                    plt.title("L-(l)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_gt_batch['last_l'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,6)
                    plt.axis('off')
                    plt.title("L-(r)")
                    plt.imshow(convert_depth_to_disp(depth=depth_sep_rendered_gt_batch['last_r'].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.savefig(os.path.join(sub_folder_GT,bin_token_list[0][:-4]+"_depth_GT.png"),bbox_inches='tight')
    
    
    
    
    if not args.output_vis:
        saved_into_json(data_dict=output_meter_into_dict(meter_rendered_results_omni_dict),
                        path = os.path.join(sub_folder_omni,"metric.json"))    
        
        saved_into_json(data_dict=output_meter_into_dict(meter_rendered_results_pixel_dict),
                        path = os.path.join(sub_folder_pixel,"metric.json"))
        
        saved_into_json(data_dict=output_meter_into_dict(meter_rendered_results_volume_dict),
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
