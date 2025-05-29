import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from diffusers.optimization import get_scheduler
import math

import depthsplat.src.datasets_stereo_matching.KITTI360.dataloader as datasets
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
from depthsplat.src.models.model_warpper import ModelWarpper
from depthsplat.src.models.encoder.unimatch.mv_unimatch  import MultiViewUniMatch


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
            project_name='depthsplat-gs', 
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


    '''Define the Dataloader Here'''
    dataset = getattr(datasets, dataset_config.dataset_name)

        
    train_params = {
        "datapath":dataset_config.datapath,
        "train_filelist":dataset_config.train_filelist,
        "val_filelist":dataset_config.val_filelist,
        "test_filelist":dataset_config.val_filelist,
        "resolution":dataset_config.resolution, 
        "split":"train",
        "use_projected_lidar":dataset_config.use_projected_lidar,
        "use_pseudo_depth":dataset_config.use_pseudo_depth }
    

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
    

    '''Define the network here, here is the depth estimatro only.'''
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
    
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    if logger is not None:
        logger.info(f'Number of params: {n_parameters}')
    
    ''' Define the optimizers '''
    # 假设 model 已经构建好
    param_groups = [
        {"params": [], "lr": 2e-4},             # 默认组
        {"params": [], "lr": 2e-4 * 0.01},      # 'pretrained' 组，lr_mult=0.01
    ]
    
    for name, param in depth_estimator_unimatch.named_parameters():
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


    ###########################################-----------------############################################################


    # move to the accelerate
    my_model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        my_model, optimizer, train_dataloader, val_dataloader, scheduler)
    

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
    print("Begin Training the Depth Estimator...")
    
    while epoch < max_num_epochs:
        my_model.train()
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, batch in enumerate(train_dataloader):
            # forward + backward + optimize
            data_time_e = time.time()
            with accelerator.accumulate(my_model):
                optimizer.zero_grad()
                try:
                    loss,log,result_dicts = my_model(batch, "train", iter=global_iter, cfg=cfg)
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue  # 直接跳过当前 iteration
                    
                    with torch.autograd.detect_anomaly():
                        accelerator.backward(loss)
        
                except:
                    torch.cuda.empty_cache()
                    print(batch['bin_token'])
                    print("Here is Error Encounter......")
                    continue  # 或 return / break

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

                # TODO here
                if global_iter % cfg.val_freq ==0:
                    my_model.eval()
                    if accelerator.is_main_process:
                        MAE_Meter = 0
                        MSE_Meter = 0
                        for i_iter_val, batch_val in enumerate(val_dataloader):
                            
                            with torch.no_grad():
                                pred_depth_val = my_model(batch_val, "val", iter=global_iter, cfg=cfg)
                                psuedo_gt_depth_val = batch_val['pseudo_depths']
                                
    
                            
                            
                            current_mae,current_mse = compute_depth_mae_mse(depth_pred=pred_depth_val,
                                                                            depth_gt=psuedo_gt_depth_val)
                            
                            MAE_Meter+=current_mae
                            MSE_Meter+=current_mse
                            
                            if i_iter_val%5==0:
                                print("Processed {}/{}".format(i_iter_val,len(val_dataloader)))
                                logger.info('MAE is {}, MSE is {}'.format(current_mae,current_mse))
 
                        MAE_Meter = MAE_Meter/(i_iter_val+1)
                        MSE_Meter = MSE_Meter/(i_iter_val+1)
                        val_batch_save_dir = osp.join(cfg.output_dir, cfg.exp_name, "validation",
                                                "step-{}/batch-{}".format(global_iter, i_iter_val))
                        os.makedirs(val_batch_save_dir,exist_ok=True)
                        
                        saved_val_mae_mse_batch = os.path.join(val_batch_save_dir,
                                                               'pred.json')
                        saved_val_results = dict()
                        saved_val_results['mae'] = MAE_Meter 
                        saved_val_results['mse'] = MSE_Meter
                        
                        
                        save_dict_to_json(data_dict=saved_val_results,
                                          save_path=saved_val_mae_mse_batch)
                        
                        
                    my_model.train()
                    



        
            time_e = time.time()
            # print loss log regularly
            if global_iter % print_freq == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]['lr']
                losses_str = ""
                for loss_k, loss_v in log.items():
                    losses_str += ("%s: %.3f, " % (loss_k, loss_v))
                if logger is not None:
                    logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f, %s grad_norm: %.1f, lr: %.7f, time: %.3f (%.3f)'%(
                        epoch, i_iter, len(train_dataloader), 
                        loss.item(), losses_str, grad_norm, lr,
                        time_e - time_s, data_time_e - data_time_s
                    ))
            
            global_iter += 1
    
            # dump loss log to tensorboard
            accelerator.log(log, step=global_iter)

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
