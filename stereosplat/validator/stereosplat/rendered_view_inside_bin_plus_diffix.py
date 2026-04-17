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
from pathlib import Path
# Runnable from any cwd:
# - `stereosplat/tools/` (helper module) under stereosplat root
# - `stereosplat/difix3d/src/` (so we can `import difix3d` without pip-installing it)
_STEREOSPLAT_ROOT = Path(__file__).resolve().parents[2]  # .../stereosplat
_DIFIX3D_SRC = _STEREOSPLAT_ROOT / "difix3d" / "src"
sys.path.insert(0, str(_STEREOSPLAT_ROOT))
if _DIFIX3D_SRC.is_dir():
    sys.path.insert(0, str(_DIFIX3D_SRC))

torch.autograd.set_detect_anomaly(True)
import numpy as np
from torch import Tensor, nn
from tools.metrics import RGB_Quality_Meter, Depth_Quality_Meter, saved_into_json
from mmengine.registry import MODELS
import json
import importlib

# define the models
from stereosplat.models_lab.StereoSplat.stereosplat import StereoSplat
from difix3d import DifixRef

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def _maybe_init_wandb(accelerator: Accelerator, args, cfg) -> bool:
    tracker_enabled = bool(getattr(args, "use_wandb", False))
    if not (tracker_enabled and accelerator.is_main_process):
        return False

    wandb_project = getattr(args, "wandb_project", None) or "StereoSplat"
    wandb_entity = getattr(args, "wandb_entity", None)
    wandb_mode = getattr(args, "wandb_mode", None)
    wandb_run_name = getattr(args, "wandb_run_name", None) or getattr(cfg, "exp_name", "validation")

    wandb_api_key = getattr(args, "wandb_api_key", None)
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    accelerator.init_trackers(
        project_name=wandb_project,
        init_kwargs={
            "wandb": {
                "name": wandb_run_name,
                **({"entity": wandb_entity} if wandb_entity else {}),
                **({"mode": wandb_mode} if wandb_mode else {}),
            }
        },
    )
    return True

def _dataset_module_for_world_center(world_center: str | None) -> str:
    if world_center is None or world_center == "Center_LiDAR":
        return "stereosplat.data.KITTI360_CenterCam_Ref.dataloader"
    if world_center == "First_Cam0":
        return "stereosplat.data.KITTI360_FirstCam_Ref.dataloader"
    if world_center == "First_LiDAR":
        return "stereosplat.data.KITTI360_FirstLiDAR_Ref.dataloader"
    if world_center == "First_LiDAR_3_Uniform":
        return "stereosplat.data.KITTI360_FisrtLiDAR_Random.dataloader"
    return "stereosplat.data.KITTI360_CenterCam_Ref.dataloader"

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
    cfg.prompt = args.prompt
    cfg.use_diffix3d_postprocessing = args.use_diffix3d_postprocessing
    logger_mm = MMLogger.get_instance('mmengine', log_level='WARNING')
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.work_dir, 
        logging_dir=os.path.join(cfg.work_dir, 'logs')
    )
    tracker_enabled = bool(getattr(args, "use_wandb", False))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=("wandb" if tracker_enabled else None),
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )
    tracker_enabled = _maybe_init_wandb(accelerator, args, cfg)

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)
    
    dataset_config = cfg.dataset_params
    
    # configure logger
    if accelerator.is_main_process:
        os.makedirs(args.output_folder, exist_ok=True)
        cfg.dump(osp.join(args.output_folder, osp.basename(args.config_path)))
    
    datasets = importlib.import_module(
        _dataset_module_for_world_center(getattr(cfg, "world_center", None))
    )

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


    # Define the Model Here
    my_model = StereoSplat(backbone=cfg.model.backbone,
                           neck=cfg.model.neck,
                           costvolume_gs=cfg.model.costvolume_gs,
                           volume_gs=cfg.model.volume_gs,
                           losses_params=cfg.model.losses_params,
                           camera_args=cfg.camera_args,
                           dataset_params=cfg.dataset_params,
                           use_checkpoint=cfg.use_checkpoint)
    
    # loading the pretrained diffix3d models.
    assert os.path.exists(args.pretrained_diffix_model_path), "The pretrained diffix3d model path does not exist!"
     
    
    pretrained_diffix_model = DifixRef(
        pretrained_name="nvidia/difix_ref",
        pretrained_path=args.pretrained_diffix_model_path,
        timestep=args.timestep,
        mv_unet=args.use_ref,
        deterministic_vae_encode=args.deterministic_vae_encode,
        deterministic_scheduler_step=args.deterministic_scheduler_step,
    )
    
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
        print("Can't find checkpoint. Randomly initialize model parameters anyway.")
        
    pretrained_diffix_model.to(accelerator.device)
    

    # performance metrics for the rendered RGBs
    evaluate_results_average_dict_rgb = {
        "first_view_psnr_average": 0,
        "first_view_ssim_average":0,
        "first_view_lpips_average":0,
        "center_view_psnr_average": 0,
        "center_view_ssim_average": 0,
        "center_view_lpips_average": 0,
        "last_view_psnr_average": 0,
        "last_view_ssim_average": 0,
        "last_view_lpips_average": 0,
        "all_view_psnr_average": 0,
        "all_view_ssim_average": 0,
        "all_view_lpips_average": 0,
    }
    
    # performance metrics for the rendered Depths
    evaluate_results_average_dict_depth = {
        "first_view_Abs_Rel_average": 0,
        "frist_view_Sq_Rel_average": 0,
        "first_view_RMSE_log_average": 0,
        "center_view_Abs_Rel_average": 0,
        "center_view_Sq_Rel_average": 0,
        "center_view_RMSE_log_average": 0,
        "last_view_Abs_Rel_average": 0,
        "last_view_Sq_Rel_average": 0,
        "last_view_RMSE_log_average": 0,
        "all_view_Abs_Rel_average": 0,
        "all_view_Sq_Rel_average": 0,
        "all_view_RMSE_log_average": 0,
    }
    
    with torch.no_grad():
        my_model.eval()
        batch_idx = 0
        for batch in tqdm(val_dataloader):
            # process the current folder
            bin_token_list = batch['bin_token']
            evaluation_results_stat = my_model.validation_on_the_forward_views_progressive_iter_once_revised(
                                            batch,
                                            args.output_folder,
                                            bin_token_list,
                                            cfg=cfg,
                                            start_images_views=2,
                                            use_diffix3d=args.use_diffix3d,
                                            diffix3d_network=pretrained_diffix_model,
                                            use_ref=args.use_ref,
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
        if tracker_enabled and accelerator.is_main_process:
            wandb_logs = {}
            for k, v in evaluate_results_average_dict_rgb.items():
                wandb_logs[f"val/rgb/{k}"] = float(v)
            for k, v in evaluate_results_average_dict_depth.items():
                wandb_logs[f"val/depth/{k}"] = float(v)
            accelerator.log(wandb_logs, step=0)


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
    parser.add_argument('--use_diffix3d_postprocessing', action='store_true', default=False)
    parser.add_argument('--deterministic_vae_encode', action='store_true', default=False)
    parser.add_argument('--deterministic_scheduler_step', action='store_true', default=False)

    parser.add_argument(
        "--output_vis",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    ) # visualize the outputs image

    # Optional W&B logging (same style as rendered_views_inside_bin)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None, help="online | offline | disabled")
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus

    main(args)