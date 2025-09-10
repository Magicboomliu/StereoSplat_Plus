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
import numpy as np
from torch import Tensor,nn
from tools.metrics import RGB_Quality_Meter,Depth_Quality_Meter,saved_into_json
from mmengine.registry import MODELS

# define the models
from models_lab.VolumeFusion.volumefusion import VolumeFusion

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

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



def main(args):
    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    
    print(cfg)
    quit()
    
    
    
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

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name='volumefusion', 
            # config=config,
            init_kwargs={
                "wandb":{'name': cfg.exp_name},
            }
        )

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)


    if cfg.world_center is not None:
        if cfg.world_center=="Center_LiDAR":
            import data.KITTI360_CenterCam_Ref.dataloader as datasets
        elif cfg.world_center=="First_Cam0":
            import data.KITTI360_FirstCam_Ref.dataloader as datasets
        elif cfg.world_center=="First_LiDAR":
            import data.KITTI360_CenterCam_Ref.dataloader as datasets
    else:
        import data.KITTI360_CenterCam_Ref.dataloader as datasets


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
    train_params = {
        "datapath":dataset_config.datapath,
        "train_filelist":dataset_config.train_filelist,
        "val_filelist":dataset_config.val_filelist,
        "test_filelist":dataset_config.val_filelist,
        "data_version":dataset_config.data_version,
        "resolution":dataset_config.resolution, 
        "split":"train",
        "sequence":dataset_config.sequence,
        "use_center":dataset_config.use_center,
        "use_first": dataset_config.use_first,
        "use_last": dataset_config.use_last,
        "supp_view_nums": dataset_config.supp_view_nums,
        "depth_info_dict":dataset_config.depth_info_params,
        "camera_model": dataset_config.camera_model
    }

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
        "supp_view_nums": 3,
        "depth_info_dict":dataset_config.depth_info_params,
        "camera_model": dataset_config.camera_model
    }

    
    # Define the dataloader
    train_dataset = dataset(**train_params)
    val_dataset = dataset(**val_params)

    train_dataloader = DataLoader(
        train_dataset, dataset_config.batch_size_train, shuffle=True,
        num_workers=dataset_config.num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, dataset_config.batch_size_val, shuffle=False,
        num_workers=dataset_config.num_workers_val
    )
    
    
    # Define the Model/Optimizer/Schduler Here
    my_model = VolumeFusion(backbone=cfg.model.backbone,
                            neck=cfg.model.neck,
                            costvolume_gs=cfg.model.costvolume_gs,
                            volume_gs=cfg.model.volume_gs,
                            losses_params=cfg.model.losses_params,
                            camera_args=cfg.camera_args,
                            dataset_params=cfg.dataset_params,
                            use_checkpoint=cfg.use_checkpoint)
    
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    if logger is not None:
        logger.info(f'Number of params: {n_parameters}')
    
    param_groups = [
        {"params": [], "lr": cfg.lr},             # 默认组
        {"params": [], "lr": cfg.lr * 0.01},      # 'pretrained' 组，lr_mult=0.01
    ]
    
    for name, param in my_model.named_parameters():
        if not param.requires_grad:
            continue
        if "pretrained" in name:
            param_groups[1]["params"].append(param)
        else:
            param_groups[0]["params"].append(param)


    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.optimizer.weight_decay,betas=(0.9, 0.999))
    # learning rate scheme
    warm_up = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        1 / (cfg.warmup_steps*accelerator.num_processes),
        1,
        total_iters=cfg.warmup_steps*accelerator.num_processes,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_train_steps*accelerator.num_processes, eta_min=cfg.lr * 0.1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, scheduler], milestones=[cfg.warmup_steps*accelerator.num_processes])


    # move to the accelerate
    my_model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        my_model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)

    # resume and load
    epoch = 0
    global_iter = 0
    first_epoch = 0


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
        global_iter = int(path.split("/")[-2].split("-")[1])
        first_epoch = global_iter // num_update_steps_per_epoch
        resume_step = global_iter % num_update_steps_per_epoch
        if accelerator.is_main_process:
            print(f'successfully resumed from epoch{first_epoch}-iter{global_iter}')
    
    else:
        resume_step = -1
    
    if accelerator.is_main_process:
        print('work dir: ', args.work_dir)
        print("max iteration steps: ",cfg.max_train_steps)

    # training along the iterations.
    print_freq = cfg.print_freq
    while epoch < max_num_epochs:
        my_model.train()
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, batch in enumerate(train_dataloader):
            data_time_e = time.time()
            
            
            with accelerator.accumulate(my_model):
                optimizer.zero_grad()
                
                try:
                    if args.gpus <= 1:
                        loss, logs,rendered_fusion_list,rendered_volume_list,rendered_cv_results_list = my_model.forward(batch, "train", iter=global_iter, cfg=cfg)
                    else:
                        loss, logs,rendered_fusion_list,rendered_volume_list,rendered_cv_results_list= my_model.module.forward(batch, "train", iter=global_iter, cfg=cfg)

        
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[Warning] NaN or INF loss at iter {global_iter}, skipping...")
                        continue  # 跳过当前 batch

                    with torch.autograd.detect_anomaly():
                        accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)

                    optimizer.step()
                    scheduler.step()
            
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"[OOM] Skipping iteration {global_iter} due to CUDA OOM.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e  # 其他错误照常抛出
                

            # Checks if the accelerator has performed an optimization step behind the scenes
            accelerator.wait_for_everyone()
            if accelerator.sync_gradients and accelerator.is_main_process:
                if global_iter % cfg.save_freq == 0:
                    if accelerator.is_main_process:
                        save_file_name = os.path.join(os.path.abspath(args.work_dir), f'checkpoint-{global_iter}')
                        accelerator.save_state(save_file_name)
                        dst_file = osp.join(args.work_dir, 'latest')
                        mmengine.utils.symlink(save_file_name, dst_file)
                        if logger is not None:
                            logger.info('[TRAIN] Save latest state dict to {}.'.format(save_file_name))
                
                
                if global_iter % 100 == 0:
                    torch.cuda.empty_cache()



            # perform the validation scripts
                if global_iter % cfg.val_freq == 0:
                    my_model.eval()
                    if accelerator.is_main_process:
                        
                        
                        #=========================================================================#
                        output_rgb_meter_center_fusion = {"left":RGB_Quality_Meter(psnr=0.0,ssim=0.0),
                                                   "right":RGB_Quality_Meter(psnr=0.0,ssim=0.0)}
                        output_rgb_meter_first_fusion = {"left":RGB_Quality_Meter(psnr=0.0,ssim=0.0),
                                                  "right":RGB_Quality_Meter(psnr=0.0,ssim=0.0)}
                        output_rgb_meter_last_fusion ={"left":RGB_Quality_Meter(psnr=0.0,ssim=0.0),
                                                "right":RGB_Quality_Meter(psnr=0.0,ssim=0.0)}
                        output_depth_meter_center_fusion = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                                    "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
                        output_depth_meter_first_fusion = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                                    "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
                        output_depth_meter_last_fusion = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                                    "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
                        input_depth_meter_fusion = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                                    "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
                        #=========================================================================#
                        output_rgb_meter_center_vol = {"left":RGB_Quality_Meter(psnr=0.0,ssim=0.0),
                                                   "right":RGB_Quality_Meter(psnr=0.0,ssim=0.0)}
                        output_rgb_meter_first_vol = {"left":RGB_Quality_Meter(psnr=0.0,ssim=0.0),
                                                  "right":RGB_Quality_Meter(psnr=0.0,ssim=0.0)}
                        output_rgb_meter_last_vol ={"left":RGB_Quality_Meter(psnr=0.0,ssim=0.0),
                                                "right":RGB_Quality_Meter(psnr=0.0,ssim=0.0)}
                        output_depth_meter_center_vol = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                                    "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
                        output_depth_meter_first_vol = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                                    "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
                        output_depth_meter_last_vol = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                                    "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
                        input_depth_meter_vol = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                                    "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
                        #=========================================================================#
                        output_rgb_meter_center_cv = {"left":RGB_Quality_Meter(psnr=0.0,ssim=0.0),
                                                   "right":RGB_Quality_Meter(psnr=0.0,ssim=0.0)}
                        output_rgb_meter_first_cv = {"left":RGB_Quality_Meter(psnr=0.0,ssim=0.0),
                                                  "right":RGB_Quality_Meter(psnr=0.0,ssim=0.0)}
                        output_rgb_meter_last_cv ={"left":RGB_Quality_Meter(psnr=0.0,ssim=0.0),
                                                "right":RGB_Quality_Meter(psnr=0.0,ssim=0.0)}
                        output_depth_meter_center_cv = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                                    "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
                        output_depth_meter_first_cv = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                                    "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
                        output_depth_meter_last_cv = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                                    "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
                        input_depth_meter_cv = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                                    "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
            
                        
                        for i_iter_val, batch_val in enumerate(val_dataloader):
                            print("Processed {}/{}".format(i_iter_val,len(val_dataloader)))
                            overall_val_batch_save_dir = osp.join(cfg.output_dir, cfg.exp_name, "validation",
                                                                  "step-{}".format(global_iter))
                            os.makedirs(overall_val_batch_save_dir,exist_ok=True)
                            val_batch_save_dir = os.path.join(overall_val_batch_save_dir,"batch-{}".format(i_iter_val))
                            os.makedirs(val_batch_save_dir,exist_ok=True)
                
                            
                            if args.gpus<=1:
                                # forward here 
                                # get the psnr, ssim, mae and mse as well as the saved the visualization results
                                metrics_rendered_rgb_list,metrics_rendered_depth_list,metrics_estimated_depth_list = my_model.validation_step(batch_val, val_batch_save_dir,cfg)
                            else:
                                metrics_rendered_rgb_list,metrics_rendered_depth_list,metrics_estimated_depth_list = my_model.module.validation_step(batch_val, val_batch_save_dir,cfg)

                            for i in range(len(metrics_rendered_rgb_list)):
                                output_rgb_meter_dict = metrics_rendered_rgb_list[i]
                                output_depth_meter_dict = metrics_rendered_depth_list[i]
                                input_depth_meter_dict = metrics_estimated_depth_list[i]
                                
      
                                
                                # saved the intermeidate results here for the RGB Here
                                output_rgb_meter_center_view_left_psnr = output_rgb_meter_dict['center_view']['left']['psnr']
                                output_rgb_meter_center_view_left_ssim = output_rgb_meter_dict['center_view']['left']['ssim']
                                output_rgb_meter_center_view_right_psnr = output_rgb_meter_dict['center_view']['right']['psnr']
                                output_rgb_meter_center_view_right_ssim = output_rgb_meter_dict['center_view']['right']['ssim']
                                
                                output_rgb_meter_first_view_left_psnr = output_rgb_meter_dict['first_view']['left']['psnr']
                                output_rgb_meter_first_view_left_ssim = output_rgb_meter_dict['first_view']['left']['ssim']   
                                output_rgb_meter_first_view_right_psnr = output_rgb_meter_dict['first_view']['right']['psnr']
                                output_rgb_meter_first_view_right_ssim = output_rgb_meter_dict['first_view']['right']['ssim']

                                output_rgb_meter_last_view_left_psnr = output_rgb_meter_dict['last_view']['left']['psnr']
                                output_rgb_meter_last_view_left_ssim = output_rgb_meter_dict['last_view']['left']['ssim']   
                                output_rgb_meter_last_view_right_psnr = output_rgb_meter_dict['last_view']['right']['psnr']
                                output_rgb_meter_last_view_right_ssim = output_rgb_meter_dict['last_view']['right']['ssim']

                                output_depth_meter_center_view_left_mae = output_depth_meter_dict['center_view']['left']['mae']
                                output_depth_meter_center_view_left_mse = output_depth_meter_dict['center_view']['left']['mse']
                                output_depth_meter_center_view_right_mae = output_depth_meter_dict['center_view']['right']['mae']
                                output_depth_meter_center_view_right_mse = output_depth_meter_dict['center_view']['right']['mse']

                                output_depth_meter_first_view_left_mae = output_depth_meter_dict['first_view']['left']['mae']
                                output_depth_meter_first_view_left_mse = output_depth_meter_dict['first_view']['left']['mse']
                                output_depth_meter_first_view_right_mae = output_depth_meter_dict['first_view']['right']['mae']
                                output_depth_meter_first_view_right_mse = output_depth_meter_dict['first_view']['right']['mse']

                                output_depth_meter_last_view_left_mae = output_depth_meter_dict['last_view']['left']['mae']
                                output_depth_meter_last_view_left_mse = output_depth_meter_dict['last_view']['left']['mse']
                                output_depth_meter_last_view_right_mae = output_depth_meter_dict['last_view']['right']['mae']
                                output_depth_meter_last_view_right_mse = output_depth_meter_dict['last_view']['right']['mse']

                                input_depth_meter_left_mae = input_depth_meter_dict['input_depth']['left']['mae']
                                input_depth_meter_left_mse = input_depth_meter_dict['input_depth']['left']['mse']
                                input_depth_meter_right_mae = input_depth_meter_dict['input_depth']['right']['mae']
                                input_depth_meter_right_mse = input_depth_meter_dict['input_depth']['right']['mse']  
                                
                                if i==0:
                                    output_rgb_meter_center_fusion['left'].update(output_rgb_meter_center_view_left_psnr,output_rgb_meter_center_view_left_ssim)
                                    output_rgb_meter_center_fusion['right'].update(output_rgb_meter_center_view_right_psnr,output_rgb_meter_center_view_right_ssim)
                                    output_rgb_meter_first_fusion['left'].update(output_rgb_meter_first_view_left_psnr,output_rgb_meter_first_view_left_ssim)
                                    output_rgb_meter_first_fusion['right'].update(output_rgb_meter_first_view_right_psnr,output_rgb_meter_first_view_right_ssim)
                                    output_rgb_meter_last_fusion["left"].update(output_rgb_meter_last_view_left_psnr,output_rgb_meter_last_view_left_ssim)
                                    output_rgb_meter_last_fusion["right"].update(output_rgb_meter_last_view_right_psnr,output_rgb_meter_last_view_right_ssim)
                                    
                                    output_depth_meter_center_fusion["left"].update(mae=output_depth_meter_center_view_left_mae,
                                                                            mse=output_depth_meter_center_view_left_mse)
                                    output_depth_meter_center_fusion["right"].update(mae=output_depth_meter_center_view_right_mae,
                                                                            mse=output_depth_meter_center_view_right_mse)
                                    output_depth_meter_first_fusion["left"].update(mae=output_depth_meter_first_view_left_mae,
                                                                            mse=output_depth_meter_first_view_left_mse)
                                    output_depth_meter_first_fusion["right"].update(mae=output_depth_meter_first_view_right_mae,
                                                                            mse=output_depth_meter_first_view_right_mse)
                                    output_depth_meter_last_fusion["left"].update(mae=output_depth_meter_last_view_left_mae,
                                                                            mse=output_depth_meter_last_view_left_mse)
                                    output_depth_meter_last_fusion["right"].update(mae=output_depth_meter_last_view_right_mae,
                                                                            mse=output_depth_meter_last_view_right_mse)
                                    input_depth_meter_fusion['left'].update(mae=input_depth_meter_left_mae,
                                                                    mse=input_depth_meter_left_mse)
                                    input_depth_meter_fusion['right'].update(mae=input_depth_meter_right_mae,
                                                                    mse=input_depth_meter_right_mse)
                                if i==1:
                                    output_rgb_meter_center_vol['left'].update(output_rgb_meter_center_view_left_psnr,output_rgb_meter_center_view_left_ssim)
                                    output_rgb_meter_center_vol['right'].update(output_rgb_meter_center_view_right_psnr,output_rgb_meter_center_view_right_ssim)
                                    output_rgb_meter_first_vol['left'].update(output_rgb_meter_first_view_left_psnr,output_rgb_meter_first_view_left_ssim)
                                    output_rgb_meter_first_vol['right'].update(output_rgb_meter_first_view_right_psnr,output_rgb_meter_first_view_right_ssim)
                                    output_rgb_meter_last_vol["left"].update(output_rgb_meter_last_view_left_psnr,output_rgb_meter_last_view_left_ssim)
                                    output_rgb_meter_last_vol["right"].update(output_rgb_meter_last_view_right_psnr,output_rgb_meter_last_view_right_ssim)
                                    
                                    output_depth_meter_center_vol["left"].update(mae=output_depth_meter_center_view_left_mae,
                                                                            mse=output_depth_meter_center_view_left_mse)
                                    output_depth_meter_center_vol["right"].update(mae=output_depth_meter_center_view_right_mae,
                                                                            mse=output_depth_meter_center_view_right_mse)
                                    output_depth_meter_first_vol["left"].update(mae=output_depth_meter_first_view_left_mae,
                                                                            mse=output_depth_meter_first_view_left_mse)
                                    output_depth_meter_first_vol["right"].update(mae=output_depth_meter_first_view_right_mae,
                                                                            mse=output_depth_meter_first_view_right_mse)
                                    output_depth_meter_last_vol["left"].update(mae=output_depth_meter_last_view_left_mae,
                                                                            mse=output_depth_meter_last_view_left_mse)
                                    output_depth_meter_last_vol["right"].update(mae=output_depth_meter_last_view_right_mae,
                                                                            mse=output_depth_meter_last_view_right_mse)
                                    input_depth_meter_vol['left'].update(mae=input_depth_meter_left_mae,
                                                                    mse=input_depth_meter_left_mse)
                                    input_depth_meter_vol['right'].update(mae=input_depth_meter_right_mae,
                                                                    mse=input_depth_meter_right_mse)
                                
                                if i==2:
                                    output_rgb_meter_center_cv['left'].update(output_rgb_meter_center_view_left_psnr,output_rgb_meter_center_view_left_ssim)
                                    output_rgb_meter_center_cv['right'].update(output_rgb_meter_center_view_right_psnr,output_rgb_meter_center_view_right_ssim)
                                    output_rgb_meter_first_cv['left'].update(output_rgb_meter_first_view_left_psnr,output_rgb_meter_first_view_left_ssim)
                                    output_rgb_meter_first_cv['right'].update(output_rgb_meter_first_view_right_psnr,output_rgb_meter_first_view_right_ssim)
                                    output_rgb_meter_last_cv["left"].update(output_rgb_meter_last_view_left_psnr,output_rgb_meter_last_view_left_ssim)
                                    output_rgb_meter_last_cv["right"].update(output_rgb_meter_last_view_right_psnr,output_rgb_meter_last_view_right_ssim)
                                    
                                    output_depth_meter_center_cv["left"].update(mae=output_depth_meter_center_view_left_mae,
                                                                            mse=output_depth_meter_center_view_left_mse)
                                    output_depth_meter_center_cv["right"].update(mae=output_depth_meter_center_view_right_mae,
                                                                            mse=output_depth_meter_center_view_right_mse)
                                    output_depth_meter_first_cv["left"].update(mae=output_depth_meter_first_view_left_mae,
                                                                            mse=output_depth_meter_first_view_left_mse)
                                    output_depth_meter_first_cv["right"].update(mae=output_depth_meter_first_view_right_mae,
                                                                            mse=output_depth_meter_first_view_right_mse)
                                    output_depth_meter_last_cv["left"].update(mae=output_depth_meter_last_view_left_mae,
                                                                            mse=output_depth_meter_last_view_left_mse)
                                    output_depth_meter_last_cv["right"].update(mae=output_depth_meter_last_view_right_mae,
                                                                            mse=output_depth_meter_last_view_right_mse)
                                    input_depth_meter_cv['left'].update(mae=input_depth_meter_left_mae,
                                                                    mse=input_depth_meter_left_mse)
                                    input_depth_meter_cv['right'].update(mae=input_depth_meter_right_mae,
                                                                    mse=input_depth_meter_right_mse)
                            
                        
                        # for this
                        output_rgb_meter_center_fusion['left'] = output_rgb_meter_center_fusion['left'].get_stats()
                        output_rgb_meter_center_fusion['right'] = output_rgb_meter_center_fusion['right'].get_stats()
                        output_rgb_meter_first_fusion['left'] = output_rgb_meter_first_fusion['left'].get_stats()
                        output_rgb_meter_first_fusion['right'] = output_rgb_meter_first_fusion['right'].get_stats()
                        output_rgb_meter_last_fusion['left'] = output_rgb_meter_last_fusion['left'].get_stats()
                        output_rgb_meter_last_fusion['right'] = output_rgb_meter_last_fusion['right'].get_stats()
                        output_depth_meter_center_fusion['left'] = output_depth_meter_center_fusion['left'].get_stats()
                        output_depth_meter_center_fusion['right'] = output_depth_meter_center_fusion['right'].get_stats()
                        output_depth_meter_first_fusion['left'] = output_depth_meter_first_fusion['left'].get_stats()
                        output_depth_meter_first_fusion['right'] = output_depth_meter_first_fusion['right'].get_stats()    
                        output_depth_meter_last_fusion['left'] = output_depth_meter_last_fusion['left'].get_stats()
                        output_depth_meter_last_fusion['right'] = output_depth_meter_last_fusion['right'].get_stats()
                        input_depth_meter_fusion['left'] = input_depth_meter_fusion['left'].get_stats()
                        input_depth_meter_fusion['right'] = input_depth_meter_fusion['right'].get_stats()

                        output_rgb_meter_center_vol['left'] = output_rgb_meter_center_vol['left'].get_stats()
                        output_rgb_meter_center_vol['right'] = output_rgb_meter_center_vol['right'].get_stats()
                        output_rgb_meter_first_vol['left'] = output_rgb_meter_first_vol['left'].get_stats()
                        output_rgb_meter_first_vol['right'] = output_rgb_meter_first_vol['right'].get_stats()
                        output_rgb_meter_last_vol['left'] = output_rgb_meter_last_vol['left'].get_stats()
                        output_rgb_meter_last_vol['right'] = output_rgb_meter_last_vol['right'].get_stats()
                        output_depth_meter_center_vol['left'] = output_depth_meter_center_vol['left'].get_stats()
                        output_depth_meter_center_vol['right'] = output_depth_meter_center_vol['right'].get_stats()
                        output_depth_meter_first_vol['left'] = output_depth_meter_first_vol['left'].get_stats()
                        output_depth_meter_first_vol['right'] = output_depth_meter_first_vol['right'].get_stats()    
                        output_depth_meter_last_vol['left'] = output_depth_meter_last_vol['left'].get_stats()
                        output_depth_meter_last_vol['right'] = output_depth_meter_last_vol['right'].get_stats()
                        input_depth_meter_vol['left'] = input_depth_meter_vol['left'].get_stats()
                        input_depth_meter_vol['right'] = input_depth_meter_vol['right'].get_stats()
                        


                        output_rgb_meter_center_cv['left'] = output_rgb_meter_center_cv['left'].get_stats()
                        output_rgb_meter_center_cv['right'] = output_rgb_meter_center_cv['right'].get_stats()
                        output_rgb_meter_first_cv['left'] = output_rgb_meter_first_cv['left'].get_stats()
                        output_rgb_meter_first_cv['right'] = output_rgb_meter_first_cv['right'].get_stats()
                        output_rgb_meter_last_cv['left'] = output_rgb_meter_last_cv['left'].get_stats()
                        output_rgb_meter_last_cv['right'] = output_rgb_meter_last_cv['right'].get_stats()
                        output_depth_meter_center_cv['left'] = output_depth_meter_center_cv['left'].get_stats()
                        output_depth_meter_center_cv['right'] = output_depth_meter_center_cv['right'].get_stats()
                        output_depth_meter_first_cv['left'] = output_depth_meter_first_cv['left'].get_stats()
                        output_depth_meter_first_cv['right'] = output_depth_meter_first_cv['right'].get_stats()    
                        output_depth_meter_last_cv['left'] = output_depth_meter_last_cv['left'].get_stats()
                        output_depth_meter_last_cv['right'] = output_depth_meter_last_cv['right'].get_stats()
                        input_depth_meter_cv['left'] = input_depth_meter_cv['left'].get_stats()
                        input_depth_meter_cv['right'] = input_depth_meter_cv['right'].get_stats()


                        results_dict_fusion = {
                            "rgb_center": output_rgb_meter_center_fusion,
                            "rgb_first": output_rgb_meter_first_fusion,
                            "rgb_last": output_rgb_meter_last_fusion,
                            "depth_center":output_depth_meter_center_fusion,
                            "depth_first":output_depth_meter_first_fusion,
                            "depth_last":output_depth_meter_last_fusion,
                            "input_depth":input_depth_meter_fusion
                            
                        }
                        results_dict_volume = {
                            "rgb_center": output_rgb_meter_center_vol,
                            "rgb_first": output_rgb_meter_first_vol,
                            "rgb_last": output_rgb_meter_last_vol,
                            "depth_center":output_depth_meter_center_vol,
                            "depth_first":output_depth_meter_first_vol,
                            "depth_last":output_depth_meter_last_vol,
                            "input_depth":input_depth_meter_vol
                            
                        }
    
                        results_dict_cv = {
                            "rgb_center": output_rgb_meter_center_cv,
                            "rgb_first": output_rgb_meter_first_cv,
                            "rgb_last": output_rgb_meter_last_cv,
                            "depth_center":output_depth_meter_center_cv,
                            "depth_first":output_depth_meter_first_cv,
                            "depth_last":output_depth_meter_last_cv,
                            "input_depth":input_depth_meter_cv
                            
                        }
                        
                        saved_into_json(data_dict=results_dict_fusion,
                                        path=os.path.join(overall_val_batch_save_dir,"fusion_metric.json"))

                        saved_into_json(data_dict=results_dict_volume,
                                        path=os.path.join(overall_val_batch_save_dir,"volume_metric.json"))

                        saved_into_json(data_dict=results_dict_cv,
                                        path=os.path.join(overall_val_batch_save_dir,"cv_metric.json"))
                        
                    my_model.train()


            time_e = time.time()

            # print loss log regularly
            if global_iter % print_freq == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]['lr']
                losses_str = ""
                for loss_k, loss_v in logs.items():
                    losses_str += ("%s: %.3f, " % (loss_k, loss_v))
                if logger is not None:
                    logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f, %s grad_norm: %.1f, lr: %.7f, time: %.3f (%.3f)'%(
                        epoch, i_iter, len(train_dataloader), 
                        loss.item(), losses_str, grad_norm, lr,
                        time_e - time_s, data_time_e - data_time_s
                    ))

            global_iter += 1

            # dump loss log to tensorboard
            accelerator.log(logs, step=global_iter)

            data_time_s = time.time()
            time_s = time.time()

        epoch += 1

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    
            
            

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config')
    parser.add_argument('--work-dir', type=str)
    parser.add_argument('--resume-from', type=str, default='')
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    
    
    main(args)