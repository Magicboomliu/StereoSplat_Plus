import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from diffusers.optimization import get_scheduler
import math



from my_custom_vggt import VGGT

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
import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
import sys
sys.path.append("../../..")
import data.KITTI360_VGGT.dataloader as datasets
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import matplotlib.pyplot as plt
import json


def saved_into_json(data_dict,path):
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=4)

class Pose_Quality_Meter(object):
    def __init__(self,rra,rta):
        self.rra = rra
        self.rta = rta
        self.counter = 0
        
    def update(self,rra,rta):
        self.rra +=rra
        self.rta +=rta
        self.counter = self.counter +1
        
    def get_stats(self):
        if self.counter==0:
            return {"rra":0,
                    "rta":0}
        else:
            return {
                "rra": self.rra * 1.0 / self.counter,
                "rta": self.rta * 1.0 /self.counter
            }

class Depth_Quality_Meter(object):
    def __init__(self,mae,mse):
        self.mae = mae
        self.mse = mse
        self.counter =0
    
    def update(self,mae,mse):
        self.mae +=mae
        self.mse +=mse
        self.counter +=1
        
    def get_stats(self):
        if self.counter==0:
            return {"mae":0,
                    "mse":0}
        else:
            return {
                "mae": self.mae * 1.0 / self.counter,
                "mse": self.mse * 1.0 /self.counter
            }

class Pcd_Quality_Meter(object):
    def __init__(self,mae,mse):
        self.mae = mae
        self.mse = mse
        self.counter =0
    
    def update(self,mae,mse):
        self.mae +=mae
        self.mse +=mse
        self.counter +=1
        
    def get_stats(self):
        if self.counter==0:
            return {"mae":0,
                    "mse":0}
        else:
            return {
                "mae": self.mae * 1.0 / self.counter,
                "mse": self.mse * 1.0 /self.counter
            }



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
            project_name='depth-gs', 
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
    

    dataset = getattr(datasets, dataset_config.dataset_name)

    train_params = {
        "datapath":dataset_config.datapath,
        "train_filelist":dataset_config.train_filelist,
        "val_filelist":dataset_config.val_filelist,
        
        "data_version":dataset_config.data_version,
        "resolution":dataset_config.resolution, 
        
        "split":"train",
        
        "sequence":dataset_config.sequence,
        "depth_info_dict":dataset_config.depth_info_params,
        "camera_model": dataset_config.camera_model,
        
        "input_type":dataset_config.input_type,
        "max_input_views":dataset_config.max_input_views,
        "pair_images":dataset_config.pair_images,
        "names_of_frames":dataset_config.names_of_frames
    
    }

    val_params = {
        "datapath":dataset_config.datapath,
        "train_filelist":dataset_config.train_filelist,
        "val_filelist":dataset_config.val_filelist,

        "data_version":dataset_config.data_version,
        "resolution":dataset_config.resolution, 
        "split":"val",
        "sequence":dataset_config.sequence,

        "depth_info_dict":dataset_config.depth_info_params,
        "camera_model": dataset_config.camera_model,

        "input_type":dataset_config.input_type,
        "max_input_views":dataset_config.max_input_views,
        "pair_images":dataset_config.pair_images,
        "names_of_frames":dataset_config.names_of_frames
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
    
    # VGGT Networks
    my_model = VGGT()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    if logger is not None:
        logger.info(f'Number of params: {n_parameters}')
    param_groups = [
        {"params": [], "lr": cfg.lr},             # 默认组
        {"params": [], "lr": cfg.lr * 0.1},      # 'pretrained' 组，lr_mult=0.01
    ]
    
    for name, param in my_model.named_parameters():
        if not param.requires_grad:
            continue
        if "aggregator" in name:
            param_groups[1]["params"].append(param)
        else:
            param_groups[0]["params"].append(param)


    if cfg.vggt_pretrained_weight!="None":
        # loaded the pretrained weight
        my_model.load_state_dict(torch.load(cfg.vggt_pretrained_weight),strict=False)
        logging.info("Loading the VGGT Weight Successfully!")



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

    # training
    print_freq = cfg.print_freq

    while epoch < max_num_epochs:
        my_model.train()
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, batch in enumerate(train_dataloader):
            data_time_e = time.time()
        
            with accelerator.accumulate(my_model):
                optimizer.zero_grad()
                if args.gpus <= 1:
                    '''
                    #dict_keys(['pose_enc', 'depth', 'depth_conf', 
                    #               'world_points', 
                    #               'world_points_conf', 'images'])
                    '''
                    predictions,loss,logs = my_model.forward(batch,mode='train',cfg=cfg) 
                else:
                    predictions,loss,logs = my_model.module.forward(batch,mode='train',cfg=cfg)
                    
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
                
                
            if global_iter % cfg.val_freq == 0:
                my_model.eval()
                if accelerator.is_main_process:
                    pose_meter = Pose_Quality_Meter(rra=0.0,rta=0.0)
                    depth_meter = Depth_Quality_Meter(mae=0.0,mse=0.0)
                    pcd_meter = Pcd_Quality_Meter(mae=0.0,mse=0.0)


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
                            predictions,loss,loss_terms,input_dict,output_rgb_meter_dict = my_model.validation_step(batch_val, val_batch_save_dir,cfg)
                        else:
                            predictions,loss,loss_terms,input_dict,output_rgb_meter_dict = my_model.module.validation_step(batch_val, val_batch_save_dir,cfg)
                

                        pose_meter.update(rra=output_rgb_meter_dict['rra'],
                                          rta=output_rgb_meter_dict['rta'])
                        depth_meter.update(mae=output_rgb_meter_dict['depth_mae'],
                                           mse=output_rgb_meter_dict['depth_mse'])
                        pcd_meter.update(mae=output_rgb_meter_dict['pcd_mae'],
                                         mse=output_rgb_meter_dict['pcd_mse'])
                        
                    
                    pose_results_dict = pose_meter.get_stats()
                    depth_results_dict = depth_meter.get_stats()
                    pcd_results_dict = pcd_meter.get_stats()

                    results_dict = {
                    "Pose_Estimation": pose_results_dict,
                    "Depth_Estimation": depth_results_dict,
                    "PointMap Estimation": pcd_results_dict,
                        
                    }
                    
                    saved_into_json(data_dict=results_dict,
                                    path=os.path.join(overall_val_batch_save_dir,"metric.json"))
                        
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







