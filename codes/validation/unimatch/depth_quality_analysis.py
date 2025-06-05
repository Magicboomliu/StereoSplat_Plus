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
from depthsplat.src.models.encoder.unimatch.mv_unimatch  import MultiViewUniMatch
import depthsplat.src.datasets_stereo_matching.KITTI360.dataloader as datasets
import matplotlib.pyplot as plt
from tqdm import tqdm


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

def saved_into_json(data_dict,path):
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=4)

class BasicMeter(object):
    def __init__(self,depth_mae,depth_mse):
        self.depth_mae = depth_mae
        self.depth_mse = depth_mse
        self.counter = 0
    
    def update(self,depth_mae,depth_mse):
        self.depth_mae +=depth_mae
        self.depth_mse +=depth_mse
        self.counter = self.counter+1
    
    def get_stats(self):
        if self.counter==0:
            return {
               "depth_mae": 0,
               "depth_mse": 0
            }

        else:
            return {
               "depth_mae": self.depth_mae/self.counter,
               "depth_mse": self.depth_mse/self.counter 
            }

def compute_depth_mae_mse(depth_pred, depth_gt, valid_min=0.0, valid_max=150.0):
    """
    Computes MAE and MSE between predicted and GT depth maps, with optional valid range filtering.

    Args:
        depth_pred (torch.Tensor): Predicted depth, shape [B, V, H, W]
        depth_gt (torch.Tensor): Ground truth depth, shape [B, V, H, W]
        valid_min (float): minimum valid GT depth
        valid_max (float): maximum valid GT depth

    Returns:
        mae (torch.Tensor): scalar mean absolute error
        mse (torch.Tensor): scalar mean squared error
    """
    assert depth_pred.shape == depth_gt.shape, "Shape mismatch between prediction and GT"

    # Create valid mask (only use pixels with valid GT depth)
    valid_mask = (depth_gt > valid_min) & (depth_gt < valid_max)

    # Compute errors
    abs_error = torch.abs(depth_pred - depth_gt)
    sq_error = (depth_pred - depth_gt) ** 2

    # Apply mask
    abs_error = abs_error[valid_mask]
    sq_error = sq_error[valid_mask]

    # Final metrics
    mae = abs_error.mean()
    mse = sq_error.mean()

    return mae, mse

def save_dict_to_json(data_dict, save_path, overwrite=True):
    """
    Save a dictionary to a JSON file.

    Args:
        data_dict (dict): Dictionary to save
        save_path (str): Full path to the JSON file (e.g., "./results/result.json")
        overwrite (bool): Whether to overwrite the file if it exists
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not overwrite and os.path.exists(save_path):
        print(f"[Warning] File already exists: {save_path}. Skipping save.")
        return

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

    print(f"[Info] Saved dictionary to: {save_path}")





def main(args):
    
    os.makedirs(args.output_dir,exist_ok=True)
    
    # load config
    cfg = Config.fromfile(args.py_config)    
    cfg.output_dir = args.output_dir
    

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, 
        logging_dir=os.path.join(cfg.output_dir, 'logs')
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
        
    ''' Dataset / Dataloader Configuration '''
    dataset_config = cfg.dataset_params
    dataset = getattr(datasets, dataset_config.dataset_name)


    val_params = {
        "datapath":dataset_config.datapath,
        "train_filelist":dataset_config.train_filelist,
        "val_filelist":dataset_config.val_filelist,
        "test_filelist":dataset_config.val_filelist,
        "resolution":dataset_config.resolution, 
        "split":"val",
        "use_projected_lidar":True,
        "use_pseudo_depth":True
    }
    
    
    val_dataset = dataset(**val_params)
    val_dataloader = DataLoader(
        val_dataset, dataset_config.batch_size_val, shuffle=False,
        num_workers=dataset_config.num_workers_val
    )
    
    encoder_cfg = cfg.model.encoder
    depth_estimator_unimatch = MultiViewUniMatch(
            num_scales=encoder_cfg.num_scales, # default is 1
            upsample_factor=encoder_cfg.upsample_factor, # upsample factor is 4
            lowest_feature_resolution=encoder_cfg.lowest_feature_resolution, # 4
            vit_type=encoder_cfg.monodepth_vit_type, # 'vits'
            unet_channels=encoder_cfg.depth_unet_channels, # 128
            grid_sample_disable_cudnn=encoder_cfg.grid_sample_disable_cudnn, # False, Grid Sampling 
        )
    
    my_model = ModelWarpper(depth_estimator=depth_estimator_unimatch)
    ###########################################-----------------############################################################
    # move to the accelerate
    my_model,val_dataloader= accelerator.prepare(
        my_model, val_dataloader
    )
    path = args.resume_from
    accelerator.print(f"Resuming from checkpoint {path}")
    accelerator.load_state(path, map_location='cpu', strict=False)

    print('successfully resumed from {}'.format(path))
    
    
    # Final Reults 
    meter_depth_meter_dict = {
        "depth_quality": BasicMeter(depth_mae=0,depth_mse=0)
    }
    

    with torch.no_grad():
        my_model.eval()
        
        my_counter = 0
        for batch_val in tqdm(val_dataloader):
            pred_depth_val = my_model(batch_val, "val", iter=my_counter, cfg=cfg)
            
            my_counter = my_counter +1
            input_images = batch_val['imgs']
            
            psuedo_gt_depth_val = batch_val['pseudo_depths']
            sparse_gt_depth_val = batch_val['sparse_depths']
            
            # get the depth quality
            current_mae,current_mse = compute_depth_mae_mse(depth_pred=pred_depth_val,
                                                            depth_gt=sparse_gt_depth_val)
            
            # calculate the depths
            meter_depth_meter_dict['depth_quality'].update(current_mae.data.item(),current_mse.data.item())
            
            assert sparse_gt_depth_val is not None

            if args.output_vis:
                
                plt.figure(figsize=(20,10))
                plt.subplot(2,3,1)
                plt.axis('off')
                plt.title("Left Images")
                plt.imshow(input_images.squeeze(0)[0].permute(1,2,0).cpu().numpy())
                plt.subplot(2,3,2)
                plt.axis('off')
                plt.title("Est Depth")
                plt.imshow(convert_depth_to_disp(depth=pred_depth_val[0,0].cpu().numpy()))
                plt.subplot(2,3,3)
                plt.axis('off')
                plt.title("Sparse GT Depth")
                plt.imshow(convert_depth_to_disp(depth=sparse_gt_depth_val[0,0].cpu().numpy()))

                plt.subplot(2,3,4)
                plt.axis('off')
                plt.title("Left Images")
                plt.imshow(input_images.squeeze(0)[1].permute(1,2,0).cpu().numpy())
                plt.subplot(2,3,5)
                plt.axis('off')
                plt.title("Est Depth")
                plt.imshow(convert_depth_to_disp(depth=pred_depth_val[0,1].cpu().numpy()))
                plt.subplot(2,3,6)
                plt.axis('off')
                plt.title("Sparse GT Depth")
                plt.imshow(convert_depth_to_disp(depth=sparse_gt_depth_val[0,1].cpu().numpy()))
                
                saved_path = os.path.join(args.output_dir,'estimated_images')
                os.makedirs(saved_path,exist_ok=True)
                
                saved_image_path = os.path.join(saved_path,"est_{}.png".format(my_counter))
                plt.savefig(saved_image_path)
                

    
    if not args.output_vis:
        saved_dict_meter = meter_depth_meter_dict["depth_quality"].get_stats()
        saved_metric_path = os.path.join(args.output_dir,'metric.json')

        save_dict_to_json(data_dict=saved_dict_meter,
                          save_path=saved_metric_path)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--resume-from', type=str, default='')

    parser.add_argument(
        "--output_vis",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    ) # visualize the outputs images


    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    
    
    main(args)