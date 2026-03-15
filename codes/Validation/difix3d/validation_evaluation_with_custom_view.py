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
from models_lab.VolumeFusion.volumefusion_revision import VolumeFusionRevision
from models_lab.diffix3D.model import Difix
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import skimage.io

from PIL import Image

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

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
    return disparity

def kitti_colormap(disparity, maxval=-1):
	return (colored_disp*np.expand_dims((disparity>0),-1)*255).astype(np.uint8)


class Basic_RGB_Meter(object):
    def __init__(self,psnr,ssim,lpips):
        self.psnr = psnr
        self.ssim = ssim
        self.lpips = lpips
        self.counter = 0
    
    def update(self,psnr,ssim,lpips):
        self.psnr +=psnr
        self.ssim +=ssim
        self.lpips +=lpips
        self.counter = self.counter+1
    
    def get_stats(self):
        if self.counter ==0:
            return{
            "psnr": 0,
            "ssim": 0,
            "lpips": 0}
        
        else:
            return {
                "psnr": self.psnr/self.counter,
                "ssim": self.ssim/self.counter,
                "lpips": self.lpips/self.counter
            }


def main(args):
    
    cfg = Config.fromfile(args.config_path)
    cfg.work_dir = args.output_folder
    cfg.prompt = args.prompt
    
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


    # Define the Model/Optimizer/Schduler Here
    my_model = VolumeFusionRevision(backbone=cfg.model.backbone,
                                    neck=cfg.model.neck,
                                    costvolume_gs=cfg.model.costvolume_gs,
                                    volume_gs=cfg.model.volume_gs,
                                    losses_params=cfg.model.losses_params,
                                    camera_args=cfg.camera_args,
                                    dataset_params=cfg.dataset_params,
                                    use_checkpoint=cfg.use_checkpoint)
    
    # loading the pretrained diffix3d models.
    assert os.path.exists(args.pretrained_diffix_model_path), "The pretrained diffix3d model path does not exist!"

    pretrained_diffix_model = Difix(
        pretrained_name=None,
        pretrained_path=args.pretrained_diffix_model_path,
        timestep=args.timestep,
        mv_unet=args.use_ref)
    
    pretrained_diffix_model.set_eval()

    my_model, val_dataloader = accelerator.prepare(
        my_model, val_dataloader
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
        
    pretrained_diffix_model.to(accelerator.device)
    
    
    final_rgb_stats_raw = {
        "0":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "0.125":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "0.25":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "0.33":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "0.5":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "0.66":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "0.75":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "1.0":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
    }

    final_rgb_stats_enhanced = {
        "0":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "0.125":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "0.25":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "0.33":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "0.5":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "0.66":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "0.75":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
        "1.0":Basic_RGB_Meter(psnr=0,ssim=0,lpips=0),
    }
    
    saved_comparsion_images_folder = os.path.join(args.output_folder,"comparsion_images")
    os.makedirs(saved_comparsion_images_folder, exist_ok=True)

    with torch.no_grad():
        my_model.eval()
        batch_idx = 0
        for batch in tqdm(val_dataloader):
            # process the current folder
            bin_token_list = batch['bin_token']
        
            scene_name = bin_token_list[0][:-4]
            
 
            
            raw_results_stat, enhanced_results_stat, saved_images_dict = my_model.test_current_difix3d_performance(
                                            batch,
                                            args.output_folder,
                                            bin_token_list,
                                            psuedo_ratio=args.pseudo_ratio,
                                            cfg=cfg,
                                            start_images_views=2,
                                            use_diffix3d=args.use_diffix3d,
                                            diffix3d_network=pretrained_diffix_model,
                                            use_ref=args.use_ref,                                            
                                            vis=args.output_vis)

            if batch_idx%50==0:
               
                saved_comparsion_images_folder_current = os.path.join(saved_comparsion_images_folder,scene_name)
                os.makedirs(saved_comparsion_images_folder_current, exist_ok=True)
                for key in saved_images_dict.keys():
                    saved_images = saved_images_dict[key]
                    saved_images_pil = Image.fromarray(saved_images)
                    saved_images_pil.save(os.path.join(saved_comparsion_images_folder_current,f"{scene_name}_{key}.png"))
            

            for key in raw_results_stat.keys():
                final_rgb_stats_raw[key].update(raw_results_stat[key]["psnr"],raw_results_stat[key]["ssim"],raw_results_stat[key]["lpips"])
            for key in enhanced_results_stat.keys():
                final_rgb_stats_enhanced[key].update(enhanced_results_stat[key]["psnr"],enhanced_results_stat[key]["ssim"],enhanced_results_stat[key]["lpips"])
            
        
            batch_idx +=1
        
            
    
    for key in final_rgb_stats_raw.keys():
        final_rgb_stats_raw[key] = final_rgb_stats_raw[key].get_stats()
    for key in final_rgb_stats_enhanced.keys():
        final_rgb_stats_enhanced[key] = final_rgb_stats_enhanced[key].get_stats()
        
    
    saved_raw_rgb_stats_name = os.path.join(args.output_folder,"raw_rgb_stats.json")
    saved_enhanced_rgb_stats_name = os.path.join(args.output_folder,"enhanced_rgb_stats.json")
    saved_into_json(final_rgb_stats_raw,saved_raw_rgb_stats_name)
    saved_into_json(final_rgb_stats_enhanced,saved_enhanced_rgb_stats_name)



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
    
    parser.add_argument('--pretrained_diffix_model_path', type=str, default="")
    parser.add_argument('--timestep', type=int, default=199)
    parser.add_argument('--prompt', type=str, default="remove degradation")
    parser.add_argument('--use_ref', action='store_true', default=False)
    parser.add_argument('--use_diffix3d', action='store_true', default=False)

    parser.add_argument(
        "--output_vis",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    ) # visualize the outputs image
    

    parser.add_argument(
        "--pseudo_ratio",
        type=str,
        nargs="*",
        default=[],
        help="List of values, e.g. --list_arg a b c",
    )
    
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus

    args.pseudo_ratio = [float(x) for x in args.pseudo_ratio]

    
    main(args)