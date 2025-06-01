
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange, pack
import math

import sys
sys.path.append("..")
from data.KITTI360 import dataloader as datasets

import mmcv
import mmengine
import imageio
from mmengine import MMLogger
from mmengine.config import Config
import logging

import moviepy.editor as mpy
import wandb

from accelerate import Accelerator
from accelerate.utils import set_seed, convert_outputs_to_fp32, DistributedType, ProjectConfiguration
from tools.visualization import depths_to_colors

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def main(args):
    # load config
    cfg = Config.fromfile(args.py_config)
    
    cfg.output_dir = args.output_dir
    
    os.makedirs(cfg.output_dir,exist_ok=True)
    
    logger_mm = MMLogger.get_instance('mmengine', log_level='WARNING')

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

    #################### Dataset Configurration #############################
    dataset_config = cfg.dataset_params    
    # build model
    from builder import builder as model_builder
    my_model = model_builder.build(cfg.model).to(accelerator.device)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)

    # generate datasets
    dataset = getattr(datasets, dataset_config.dataset_name)

    if args.validation_list!='None':
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
        print('Successfully loaded from path {}'.format(path))
    else:
        print('Can\'t find checkpoint {}. Randomly initialize model parameters anyway.'.format(args.load_from))

    
    
    
    with torch.no_grad():
        my_model.eval()
        for batch in tqdm(val_dataloader):
            
            data_time_e = time.time()
            if torch.cuda.device_count() > 1:
                # pred results and the bin tokens   
                preds, bin_tokens = my_model.module.forward_demo_kitti360(batch)
            else:
                preds, bin_tokens = my_model.forward_demo_kitti360(batch)
        
            bs = preds["img"].shape[0]  
            pred_imgs = preds["img"] #(4,960,3,224,400)
            pred_depths = preds["depth"] #(4,960,3,224,400)
            
            

            # saved the results with batch
            for b in range(bs):
                bin_token = bin_tokens[b]
                
                # dump rgb view
                dump_path = osp.join(cfg.output_dir, "{}_rgb.mp4".format(bin_token))
                video = (pred_imgs[b].clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
                video_rec = wandb.Video(video[None], fps=30, format="mp4")
                video_tensor = video_rec._prepare_video(video_rec.data)
                clip = mpy.ImageSequenceClip(list(video_tensor), fps=30)
                clip.write_videofile(dump_path, codec='libx264', preset='medium', logger=None)
                
                # dump depth view
                dump_path_dpt = osp.join(cfg.output_dir, "{}_depth.mp4".format(bin_token))
                pred_depth = pred_depths[b].clamp(0.0, 100.0)
                max_val = float(pred_depth.max())
                video_dpt = depths_to_colors(pred_depths[b], concat="frame", max_val=max_val)
                video_dpt = video_dpt.transpose((0, 3, 1, 2))
                video_rec_dpt = wandb.Video(video_dpt[None], fps=30, format="mp4")
                video_tensor_dpt = video_rec_dpt._prepare_video(video_rec_dpt.data)
                clip_dpt = mpy.ImageSequenceClip(list(video_tensor_dpt), fps=30)
                clip_dpt.write_videofile(dump_path_dpt, codec='libx264', preset='medium', logger=None)
        
        torch.cuda.empty_cache()
        time_e = time.time()


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument("--validation_list",type=str,default='') # use larger list for evaluation
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    
    main(args)