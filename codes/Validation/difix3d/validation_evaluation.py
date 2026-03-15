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
from models_lab.diffix3D.model import Difix

import skimage.io
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
from tqdm import tqdm
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import json


def save_dict_to_json(data: dict, json_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def compute_psnr_ssim(render_rgb_path, gt_rgb_path):
    """
    Args:
        render_rgb_path: str, rendered RGB image path
        gt_rgb_path: str, ground-truth RGB image path

    Returns:
        psnr: float
        ssim: float
    """
    render = Image.open(render_rgb_path).convert("RGB")
    gt = Image.open(gt_rgb_path).convert("RGB")

    render = np.array(render).astype(np.float32) / 255.0
    gt = np.array(gt).astype(np.float32) / 255.0

    if render.shape != gt.shape:
        raise ValueError(f"Image shapes do not match: {render.shape} vs {gt.shape}")

    psnr = peak_signal_noise_ratio(gt, render, data_range=1.0)
    ssim = structural_similarity(gt, render, channel_axis=-1, data_range=1.0)

    return float(psnr), float(ssim)

def compute_psnr_ssim_from_pil(render_pil: Image.Image, gt_pil: Image.Image):
    """
    Args:
        render_pil: PIL.Image
        gt_pil: PIL.Image

    Returns:
        psnr: float
        ssim: float
    """
    render = np.array(render_pil.convert("RGB")).astype(np.float32) / 255.0
    gt = np.array(gt_pil.convert("RGB")).astype(np.float32) / 255.0

    if render.shape != gt.shape:
        raise ValueError(f"Image shapes do not match: {render.shape} vs {gt.shape}")

    mse = np.mean((render - gt) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = -10.0 * np.log10(mse)

    ssim = structural_similarity(gt, render, channel_axis=-1, data_range=1.0)

    return float(psnr), float(ssim)

class Basic_RGB_Meter(object):
    def __init__(self,psnr,ssim):
        self.psnr = psnr
        self.ssim = ssim
        self.counter = 0
    
    def update(self,psnr,ssim):
        self.psnr +=psnr
        self.ssim +=ssim
        self.counter = self.counter+1
    
    def get_stats(self):
        if self.counter ==0:
            return{
            "psnr": 0,
            "ssim": 0}
        
        else:
            return {
                "psnr": self.psnr/self.counter,
                "ssim": self.ssim/self.counter
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
    

    
    # configure logger
    if accelerator.is_main_process:
        os.makedirs(args.output_folder, exist_ok=True)
        cfg.dump(osp.join(args.output_folder, osp.basename(args.config_path)))
        
        
    # Loading the Pre-trained Difix3D Model Here
    pretrained_diffix_model = Difix(
        pretrained_name=None,
        pretrained_path=args.pretrained_diffix_model_path,
        timestep=args.timestep,
        mv_unet=args.use_ref)
    
    pretrained_diffix_model.set_eval()
    print("Successfully loading the pre-trained Difix3D Model From Path {}.....".format(args.pretrained_diffix_model_path))
    
    
    saved_json_path = os.path.join(args.output_folder, "RGB_Quality_Meter_Dict.json")
    

    saved_enhanced_rgb_folder = os.path.join(args.output_folder, "enhanced_views")
    os.makedirs(saved_enhanced_rgb_folder, exist_ok=True)

    # data source
    raw_data_folder = os.path.join(args.finetuned_difix3d_dataset_path, "rendered_views")
    assert os.path.exists(raw_data_folder), "The raw data folder does not exist"
    gt_data_folder = os.path.join(args.finetuned_difix3d_dataset_path, "gt_views")
    assert os.path.exists(gt_data_folder), "The gt data folder does not exist"
    
    reference_data_folder = os.path.join(args.finetuned_difix3d_dataset_path, "reference_views")
    assert os.path.exists(reference_data_folder), "The reference data folder does not exist"
    
    
    saved_enhanced_rgb_folder = os.path.join(args.output_folder, "enhanced_views")
    os.makedirs(saved_enhanced_rgb_folder, exist_ok=True)
    
    
    # raw psnr and ssim of center rgb left and right views.
    center_rgb_left_meter = Basic_RGB_Meter(psnr=0, ssim=0)
    center_rgb_right_meter = Basic_RGB_Meter(psnr=0, ssim=0)
    last_rgb_left_meter = Basic_RGB_Meter(psnr=0, ssim=0)
    last_rgb_right_meter = Basic_RGB_Meter(psnr=0, ssim=0)
    
    # enhanced psnr and ssim of center rgb left and right views.
    enhance_center_rgb_left_meter = Basic_RGB_Meter(psnr=0, ssim=0)
    enhance_center_rgb_right_meter = Basic_RGB_Meter(psnr=0, ssim=0)
    enhance_last_rgb_left_meter = Basic_RGB_Meter(psnr=0, ssim=0)
    enhance_last_rgb_right_meter = Basic_RGB_Meter(psnr=0, ssim=0)
    
    
    for scene_name in tqdm(sorted(os.listdir(raw_data_folder))):
        for fname in sorted(os.listdir(os.path.join(raw_data_folder, scene_name))):
            
            raw_rgb_path = os.path.join(raw_data_folder, scene_name, fname)
            gt_rgb_path = os.path.join(gt_data_folder, scene_name, fname.replace('rendered_', 'gt_'))
            
            reference_rgb_path = os.path.join(reference_data_folder, scene_name, 
                                              fname.replace('rendered_', 'reference_first_'))
            
            saved_enhanced_rgb_path = os.path.join(saved_enhanced_rgb_folder, 
                                                   scene_name, 
                                                   fname.replace('rendered_', 'enhanced_'))
            
            os.makedirs(os.path.dirname(saved_enhanced_rgb_path), exist_ok=True)

            

            assert os.path.exists(raw_rgb_path), "The raw rgb path does not exist"
            assert os.path.exists(gt_rgb_path), "The gt rgb path does not exist"
            assert os.path.exists(reference_rgb_path), "The reference rgb path does not exist"
            
            # using the pretrained difix3d here for the quality enhancement.
            with torch.no_grad():
                raw_rgb_pil = Image.open(raw_rgb_path).convert("RGB")
                gt_rgb_pil = Image.open(gt_rgb_path).convert("RGB")
                reference_rgb_pil = Image.open(reference_rgb_path).convert("RGB")
                current_width, current_height = raw_rgb_pil.size
                
                enhance_rgb_pil = pretrained_diffix_model.sample(
                                            raw_rgb_pil,
                                            height=112,
                                            width=544,
                                            ref_image=reference_rgb_pil,
                                            prompt=args.prompt)
                if not os.path.exists(saved_enhanced_rgb_path):
                    enhance_rgb_pil.save(saved_enhanced_rgb_path)
                    
                    
                
                
            # raw the psnr and ssim of the raw rgb image from the stereosplat            
            current_raw_psnr, current_raw_ssim = compute_psnr_ssim(raw_rgb_path, 
                                                                   gt_rgb_path)
            
            # the psnr and ssim of the enhanced rgb image from the stereosplat enhanced by 
            # the pretrained difix3d model.
            current_enhance_psnr, current_enhance_ssim = compute_psnr_ssim_from_pil(enhance_rgb_pil, 
                                                                                    gt_rgb_pil)
            

            
            if 'center' in fname:
                if 'left' in fname:    
                    center_rgb_left_meter.update(current_raw_psnr, current_raw_ssim)
                    enhance_center_rgb_left_meter.update(current_enhance_psnr, current_enhance_ssim)
                    
                elif 'right' in fname:
                    center_rgb_right_meter.update(current_raw_psnr, current_raw_ssim)
                    enhance_center_rgb_right_meter.update(current_enhance_psnr, current_enhance_ssim)

                    
            elif 'last' in fname:
                if 'left' in fname:
                    last_rgb_left_meter.update(current_raw_psnr, current_raw_ssim)
                    enhance_last_rgb_left_meter.update(current_enhance_psnr, current_enhance_ssim)
                elif 'right' in fname:
                    last_rgb_right_meter.update(current_raw_psnr, current_raw_ssim)
                    enhance_last_rgb_right_meter.update(current_enhance_psnr, current_enhance_ssim)
        
    
    # raw metrics
    average_center_rgb_left_metrics = center_rgb_left_meter.get_stats()
    average_center_rgb_right_metrics = center_rgb_right_meter.get_stats()
    average_last_rgb_left_metrics = last_rgb_left_meter.get_stats()
    average_last_rgb_right_metrics = last_rgb_right_meter.get_stats()
    
    
    # enhanced metrics
    enhanced_average_center_rgb_left_metrics = enhance_center_rgb_left_meter.get_stats()
    enhanced_average_center_rgb_right_metrics = enhance_center_rgb_right_meter.get_stats()
    enhanced_average_last_rgb_left_metrics = enhance_last_rgb_left_meter.get_stats()
    enhanced_average_last_rgb_right_metrics = enhance_last_rgb_right_meter.get_stats()
    
    
    
    RGB_Quality_Meter_Dict = {
        "raw_center_rgb_left": average_center_rgb_left_metrics,
        "raw_center_rgb_right": average_center_rgb_right_metrics,
        "raw_last_rgb_left": average_last_rgb_left_metrics,
        "raw_last_rgb_right": average_last_rgb_right_metrics,
        "enhanced_center_rgb_left": enhanced_average_center_rgb_left_metrics,
        "enhanced_center_rgb_right": enhanced_average_center_rgb_right_metrics,
        "enhanced_last_rgb_left": enhanced_average_last_rgb_left_metrics,
        "enhanced_last_rgb_right": enhanced_average_last_rgb_right_metrics,
    }
    
    
    save_dict_to_json(RGB_Quality_Meter_Dict, saved_json_path)
    
    



def get_mean(list):
    return sum(list)*1.0/len(list)
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path')
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--val_filelist', type=str, default='')
    parser.add_argument('--demo_filelist', type=str, default='')
    parser.add_argument('--dataset_type', type=str)
    
    
    parser.add_argument('--finetuned_difix3d_dataset_path', type=str, default='')
    parser.add_argument('--ablation_type', type=str)
    
    parser.add_argument('--pretrained_diffix_model_path', type=str, default="")
    parser.add_argument('--timestep', type=int, default=199)
    parser.add_argument('--prompt', type=str, default="remove degradation")
    parser.add_argument('--use_ref', action='store_true', default=False)

    parser.add_argument(
        "--output_vis",
        action="store_true",
        help=" something else",
    ) # visualize the outputs image
    
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus

    main(args)
