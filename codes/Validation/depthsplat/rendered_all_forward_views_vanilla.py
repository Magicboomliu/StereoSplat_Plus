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
from torch import Tensor,nn
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import set_seed, convert_outputs_to_fp32, DistributedType, ProjectConfiguration, InitProcessGroupKwargs
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import json
import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
import json

import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

import sys
sys.path.append("..")

from depthsplat.vanilla.models.encoder.unimatch.mv_unimatch import MultiViewUniMatch
from depthsplat.vanilla.models.encoder.unimatch.dpt_head import DPTHead
from depthsplat.vanilla.models.encoder.heads.gaussains_head import Gaussains_Estimator_Head,GaussianAdapterCfg
from depthsplat.vanilla.models.decoder.decoder_splatting_head_cuda import DecoderSplattingCUDA
from depthsplat.vanilla.models.model_warpper_splat import ModelWarpper
from tools.metrics import RGB_Quality_Meter,Depth_Quality_Meter,saved_into_json

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

class DecoderCFG(object):
    def __init__(self,background_color=[0.0, 0.0, 0.0]):
        self.background_color = background_color



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
        kwargs_handlers=[kwargs])

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
        # import data.KITTI360_For_Val.KITTI360_CenterCam_Ref.dataloader as datasets
        import data.KITTI360_FirstLiDAR_Ref.dataloader as datasets
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
    

    '''     Model Configuration   '''
    encoder_cfg = cfg.model.encoder
    
    # depth unimatch model
    depth_estimator_unimatch = MultiViewUniMatch(
            num_scales=encoder_cfg.num_scales, # default is 1
            upsample_factor=encoder_cfg.upsample_factor, # upsample factor is 4
            lowest_feature_resolution=encoder_cfg.lowest_feature_resolution, # 4
            vit_type=encoder_cfg.monodepth_vit_type, # 'vits'
            unet_channels=encoder_cfg.depth_unet_channels, # 128
            grid_sample_disable_cudnn=encoder_cfg.grid_sample_disable_cudnn, # False, Grid Sampling 
        )
    
    # 3dgs head: define the the gaussain head
    gaussian_adapter_config = cfg.model.encoder.gaussian_adapter
    # color branch
    gaussain_color_branch_config = {
            "large_gaussian_head": cfg.model.encoder.large_gaussian_head,
            "color_large_unet": cfg.model.encoder.color_large_unet,
            "init_sh_input_img": cfg.model.encoder.init_sh_input_img,
            "feature_upsampler_channels": cfg.model.encoder.feature_upsampler_channels,
            "gaussian_regressor_channels": cfg.model.encoder.gaussian_regressor_channels,
            "num_surfaces":cfg.model.encoder.num_surfaces}
    
    # gaussain head estimation
    gaussain_head = Gaussains_Estimator_Head(monodepth_vit_type=cfg.model.encoder.monodepth_vit_type,
                                             upsample_factor=cfg.model.encoder.upsample_factor,
                                             num_scales=cfg.model.encoder.num_scales,
                                             gaussian_head_settings_dict=gaussian_adapter_config,
                                             gaussians_color_branch_dict=gaussain_color_branch_config)
    
    dataset_config = DecoderCFG(background_color=cfg.background_color)
    depthsplattercuda_decoder = DecoderSplattingCUDA(dataset_cfg=dataset_config)
    
    
    my_model = ModelWarpper(depth_estimator=depth_estimator_unimatch,
                            gaussain_head=gaussain_head,
                            decoder_branch=depthsplattercuda_decoder,
                            unimatch_weight = cfg.unimatch_weights_path
                            )
    

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    
    print("Number of params: ",n_parameters)


    # move to the accelerate
    my_modelval_dataloader = accelerator.prepare(
        my_model,  val_dataloader
    )

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

    evaluate_results_average_dict_rgb = {
        "first_view_psnr_left": 0,
        "first_view_ssim_left": 0,
        "first_view_psnr_right": 0,
        "first_view_ssim_right": 0,
        "center_view_psnr_left": 0,
        "center_view_ssim_left": 0,
        "center_view_psnr_right": 0,
        "center_view_ssim_right": 0,
        "last_view_psnr_left": 0,
        "last_view_ssim_left": 0,
        "last_view_psnr_right": 0,
        "last_view_ssim_right": 0,
        "all_view_psnr_left": 0,
        "all_view_ssim_left": 0,
        "all_view_psnr_right": 0,
        "all_view_ssim_right": 0,

    }
    
    evaluate_results_average_dict_depth = {
        "first_view_left_mae": 0,
        "first_view_left_mse": 0,
        "first_view_right_mae": 0,
        "first_view_right_mse": 0,
        "center_view_left_mae": 0,
        "center_view_left_mse": 0,
        "center_view_right_mae": 0,
        "center_view_right_mse": 0,
        "last_view_left_mae": 0,
        "last_view_left_mse": 0,
        "last_view_right_mae": 0,
        "last_view_right_mse": 0,
        "all_view_left_mae": 0,
        "all_view_left_mse": 0,
        "all_view_right_mae": 0,
        "all_view_right_mse": 0,
    }


    with torch.no_grad():
        my_model.eval()
        batch_idx = 0
        for batch in tqdm(val_dataloader):
            # process the current folder
            bin_token_list = batch['bin_token']
            
            evaluation_results_stat =my_model.validation_complete_with_bin_tokens(batch,
                                                        args.output_folder,
                                                        bin_token_list,
                                                        cfg=cfg,
                                                        vis=args.output_vis)


            current_evaluate_results_dict_rgb = evaluation_results_stat["RGB"]
            current_evaluate_results_dict_depth = evaluation_results_stat["Depth"]
            
            for key in current_evaluate_results_dict_rgb.keys():
                evaluate_results_average_dict_rgb[key] += current_evaluate_results_dict_rgb[key]
            for key in current_evaluate_results_dict_depth.keys():
                evaluate_results_average_dict_depth[key] += current_evaluate_results_dict_depth[key]
            batch_idx += 1
           
        for key in evaluate_results_average_dict_rgb.keys():
            evaluate_results_average_dict_rgb[key] /= batch_idx
        for key in evaluate_results_average_dict_depth.keys():
            evaluate_results_average_dict_depth[key] /= batch_idx
            
        results_dict = {
            "rgb": evaluate_results_average_dict_rgb,
            "depth": evaluate_results_average_dict_depth,
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