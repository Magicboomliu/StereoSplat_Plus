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
import json

def saved_into_json(data_dict,path):
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=4)

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
    
    if args.dataset_type=="Center_LiDAR":
        import sys
        sys.path.append("..")
        import data.KITTI360_For_Val.KITTI360_CenterCam_Ref.dataloader as datasets
    elif args.dataset_type=="First_LiDAR":
        import sys
        sys.path.append("..")
        import data.KITTI360_For_Val.KITTI360_CenterCam_Ref.dataloader as datasets
    else:
        raise NotImplementedError
    
    
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
        "supp_view_nums": 3,
        "depth_info_dict":dataset_config.depth_info_params,
        "camera_model": dataset_config.camera_model,
        "pair_images": dataset_config.pair_images
    }
    
    val_dataset = dataset(**val_params)
    val_dataloader = DataLoader(
        val_dataset, dataset_config.batch_size_val, shuffle=False,
        num_workers=dataset_config.num_workers_val
    )


    from builder import builder as model_builder
    my_model = model_builder.build(cfg.model).to(accelerator.device)
    # move to the accelerate
    my_model, val_dataloader = accelerator.prepare(my_model, val_dataloader)
    
    

    # Potentially load in the weights and states from a previous save
    if args.pretrained_model_path:
        cfg.pretrained_model_path = args.pretrained_model_path
    if cfg.pretrained_model_path:
        path = cfg.pretrained_model_path
    else:
        path = None


    if path:
        accelerator.print(f"Loading from checkpoint {path}")
        accelerator.load_state(path, map_location='cpu', strict=False)
        print('Successfully loaded from {}'.format(path))
    else:
        print('Can\'t find checkpoint {}. Randomly initialize model parameters anyway.'.format(args.load_from))



    meter_rendered_results_dict = {
        "center_l": Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'center_r': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'first_l': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'first_r': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'last_l': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        'last_r': Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        "all_avg_l":Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
        "all_avg_r":Basic_Meter(psnr=0,ssim=0,mae=0,mse=0),
    }



    with torch.no_grad():
        my_model.eval()
        for batch in tqdm(val_dataloader):
            # process the current folder
            bin_token_list = batch['bin_token']
            
            rendered_left_images_list,rendered_right_images_list,rendered_left_depth_list,rendered_right_depth_list, \
            left_psnr_list,left_ssim_list,right_psnr_list,right_ssim_list,left_depth_mae_list,left_depth_mse_list,right_depth_mae_list,right_depth_mse_list= my_model.validation_complete_with_bin_tokens(batch,
                                            args.output_folder,
                                            bin_token_list,
                                            saved_label=args.output_vis
                                                         )
            
            left_psnr_first,left_psnr_center,left_psnr_last = left_psnr_list[:3]
            left_ssim_first, left_ssim_center,left_ssim_last = left_ssim_list[:3]
            right_psnr_first,right_psnr_center,right_psnr_last = right_psnr_list[:3]
            right_ssim_first,right_ssim_center,right_ssim_last = right_ssim_list[:3]
            
            left_mae_first,left_mae_center,left_mae_last = left_depth_mae_list[:3]
            left_mse_first,left_mse_center,left_mse_last = left_depth_mse_list[:3]
            
            right_mae_first,right_mae_center,right_mae_last = right_depth_mae_list[:3]
            right_mse_first,right_mse_center,right_mse_last = right_depth_mse_list[:3]
            
            
            meter_rendered_results_dict["first_l"].update(psnr=left_psnr_first,
                                                          ssim=left_ssim_first,
                                                          mae=left_mae_first,
                                                          mse=left_mse_first
                                                          )
            meter_rendered_results_dict["first_r"].update(psnr=right_psnr_first,
                                                          ssim=right_ssim_first,
                                                          mae=right_mae_first,
                                                          mse=right_mse_first
                                                          )
            
            meter_rendered_results_dict["center_l"].update(
                                                        psnr=left_psnr_center,
                                                        ssim=left_ssim_center,
                                                        mae=left_mae_center,
                                                        mse=left_mse_center)
            
            meter_rendered_results_dict["center_r"].update(
                                                        psnr=right_psnr_center,
                                                        ssim=right_ssim_center,
                                                        mae=right_mae_center,
                                                        mse=right_mse_center)
            
            meter_rendered_results_dict["last_l"].update(
                                                        psnr=left_psnr_last,
                                                        ssim=left_ssim_last,
                                                        mae=left_mae_last,
                                                        mse=left_mse_last)
            
            meter_rendered_results_dict["last_r"].update(
                                                        psnr = right_psnr_last,
                                                        ssim = right_ssim_last,
                                                        mae = right_mae_last,
                                                        mse = right_mse_last)
            
            meter_rendered_results_dict["all_avg_l"].update(
                                        psnr=get_mean(left_psnr_list),
                                        ssim =get_mean(left_ssim_list),
                                        mae = get_mean(left_depth_mae_list),
                                        mse= get_mean(left_depth_mse_list)
                )
            
            meter_rendered_results_dict["all_avg_r"].update(
                                        psnr=get_mean(right_psnr_list),
                                        ssim =get_mean(right_ssim_list),
                                        mae = get_mean(right_depth_mae_list),
                                        mse= get_mean(right_depth_mse_list)
            )
            
        
        results_dict = {
            'first_l': meter_rendered_results_dict['first_l'].get_stats(),
            'first_r': meter_rendered_results_dict['first_r'].get_stats(),
            "center_l": meter_rendered_results_dict['center_l'].get_stats(),
            'center_r': meter_rendered_results_dict['center_r'].get_stats(),
            'last_l': meter_rendered_results_dict['last_l'].get_stats(),
            'last_r': meter_rendered_results_dict['last_r'].get_stats(),
            "all_avg_l":meter_rendered_results_dict['all_avg_l'].get_stats(),
            "all_avg_r":meter_rendered_results_dict['all_avg_r'].get_stats(),
        }
        if not args.output_vis:
            saved_into_json(data_dict=results_dict,
                                path=os.path.join(args.output_folder,"metric.json"))
                        
def get_mean(list):
    return sum(list)*1.0/len(list)
    
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

    parser.add_argument(
        "--output_vis",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    ) # visualize the outputs image
    
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    main(args)