import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append("..")

from torch.utils.data import DataLoader
from einops import rearrange
from diffusers.optimization import get_scheduler
import math
import data.KITTI360.dataloader as datasets

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

from depthsplat.src.models.encoder.unimatch.mv_unimatch import MultiViewUniMatch
from depthsplat.src.models.encoder.unimatch.dpt_head import DPTHead
import numpy as np
from depthsplat.src.models.encoder.heads.gaussains_head import Gaussains_Estimator_Head,GaussianAdapterCfg
from torch import Tensor,nn
from depthsplat.src.models.decoder.decoder_splatting_head_cuda import DecoderSplattingCUDA
from depthsplat.src.models.model_warpper_splat import ModelWarpper

from tools.metrics import RGB_Quality_Meter,Depth_Quality_Meter,saved_into_json

import moviepy.editor as mpy
import wandb

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from tools.visualization import depths_to_colors



def create_logger(log_file=None, is_main_process=False, log_level=logging.INFO):
    if not is_main_process:
        return None
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if is_main_process else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if is_main_process else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if is_main_process else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger

class DecoderCFG(object):
    def __init__(self,background_color=[0.0, 0.0, 0.0]):
        self.background_color = background_color




def main(args):
    

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    
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

    if args.output_vis:
        cfg.validation_vis_progress = True
    else:
        cfg.validation_vis_progress = False
    

    #################### Dataset Configurration #############################
    dataset_config = cfg.dataset_params
    max_num_epochs = cfg.max_epochs # default is 30

    # configure logger
    if accelerator.is_main_process:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    if not osp.exists(osp.dirname(log_file)):
        os.makedirs(osp.dirname(log_file),exist_ok=True)
    logger = create_logger(log_file=log_file, is_main_process=accelerator.is_main_process)
    if logger is not None:
        logger.info(f'Config:\n{cfg.pretty_text}')


    # generate datasets
    dataset = getattr(datasets, dataset_config.dataset_name)
    val_params = {
        "datapath":dataset_config.datapath,
        "train_filelist":dataset_config.train_filelist,
        "val_filelist":args.val_filelist,
        "test_filelist":args.val_filelist,
        "data_version":dataset_config.data_version,
        "resolution":dataset_config.resolution, 
        "split":"val",
        "sequence":dataset_config.sequence,
        "use_center":dataset_config.use_center,
        "use_first": dataset_config.use_first,
        "use_last": dataset_config.use_last,
        "supp_view_nums": 3,
        "depth_info_dict":dataset_config.depth_info_params,
        "camera_model": dataset_config.camera_model
    }
    
    val_dataset = dataset(**val_params)
    val_dataloader = DataLoader(
        val_dataset, dataset_config.batch_size_val, shuffle=False,
        num_workers=dataset_config.num_workers_val
    )
    
    
    # loaded the models
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

    # move to the accelerate
    my_model, val_dataloader = accelerator.prepare(
        my_model, val_dataloader
    )


    # Potentially load in the weights and states from a previous save
    if args.resume_from:
        cfg.resume_from = args.resume_from
    if cfg.resume_from:
        if cfg.resume_from == "None":
            path = None
        elif cfg.resume_from != "latest":
            path = cfg.resume_from
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.work_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            if len(dirs) > 0:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]
            else:
                path = None

    if path:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(path, map_location='cpu', strict=False)


        if accelerator.is_main_process:
            print('successfully resumed from {}'.format(path))
    
    else:
        resume_step = -1
        print("No Pretrained Weighted Founded, Learning from Scratch")
    
    # replace here
    cfg.output_dir = args.work_dir
    
    # do the visualization here
    with torch.no_grad():
        my_model.eval()
        if accelerator.is_main_process:            
            for i_iter_val, batch_val in enumerate(val_dataloader):
                print("Processed {}/{}".format(i_iter_val,len(val_dataloader)))
                overall_val_batch_save_dir = osp.join(cfg.output_dir, cfg.exp_name, "validation")
                os.makedirs(overall_val_batch_save_dir,exist_ok=True)
                val_batch_save_dir = os.path.join(overall_val_batch_save_dir,"batch-{}".format(i_iter_val))
                os.makedirs(val_batch_save_dir,exist_ok=True)
    
                
                if args.gpus<=1:
                    # forward here 
                    # get the psnr, ssim, mae and mse as well as the saved the visualization results
                    preds, bin_tokens = my_model.forward_video_kitti360(batch_val, val_batch_save_dir,cfg)
                else:
                    preds, bin_tokens = my_model.module.forward_video_kitti360(batch_val, val_batch_save_dir,cfg)

  
                
                bs = preds["img"].shape[0]  
                pred_imgs = preds["img"] #(B,960,3,224,400)
                pred_depths = preds["depth"] #(B,960,3,224,400)

                # saved the results with batch
                for b in range(bs):
                    bin_token = bin_tokens[b][:-4]
                    
                    # dump rgb view
                    dump_path = osp.join(cfg.output_dir, "saved_videos/{}_rgb.mp4".format(bin_token))
                    os.makedirs(os.path.dirname(dump_path),exist_ok=True)
                    video = (pred_imgs[b].clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
                    video_rec = wandb.Video(video[None], fps=30, format="mp4")
                    video_tensor = video_rec._prepare_video(video_rec.data)
                    clip = mpy.ImageSequenceClip(list(video_tensor), fps=30)
                    clip.write_videofile(dump_path, codec='libx264', preset='medium', logger=None)
                    
                    # dump depth view
                    dump_path_dpt = osp.join(cfg.output_dir, "saved_videos/{}_depth.mp4".format(bin_token))
                    os.makedirs(os.path.dirname(dump_path_dpt),exist_ok=True)
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

if __name__=="__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config')
    parser.add_argument('--work-dir', type=str)
    parser.add_argument('--val_filelist', type=str)
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument(
        "--output_vis",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    ) 


    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    
    
    main(args)
