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

import matplotlib.pyplot as plt

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

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
    # max_num_epochs = cfg.max_epochs # default is 30

    # # configure logger
    # if accelerator.is_main_process:
    #     os.makedirs(args.work_dir, exist_ok=True)
    #     cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = osp.join(args.work_dir, f'{timestamp}.log')
    # if not osp.exists(osp.dirname(log_file)):
    #     os.makedirs(osp.dirname(log_file),exist_ok=True)
    # logger = create_logger(log_file=log_file, is_main_process=accelerator.is_main_process)
    # if logger is not None:
    #     logger.info(f'Config:\n{cfg.pretty_text}')


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
            output_rgb_meter_center = {"left":RGB_Quality_Meter(psnr=0.0,ssim=0.0),
                                        "right":RGB_Quality_Meter(psnr=0.0,ssim=0.0)}
            output_rgb_meter_first = {"left":RGB_Quality_Meter(psnr=0.0,ssim=0.0),
                                        "right":RGB_Quality_Meter(psnr=0.0,ssim=0.0)}
            output_rgb_meter_last ={"left":RGB_Quality_Meter(psnr=0.0,ssim=0.0),
                                    "right":RGB_Quality_Meter(psnr=0.0,ssim=0.0)}
            output_depth_meter_center = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                        "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
            output_depth_meter_first = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                        "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
            output_depth_meter_last = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                        "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
            input_depth_meter = {"left": Depth_Quality_Meter(mae=0.0,mse=0.0),
                                        "right": Depth_Quality_Meter(mae=0.0,mse=0.0)}
            
            GT_vis_output_folder = os.path.join(cfg.work_dir,"GT")
            depthsplat_rendered_output_folder = os.path.join(cfg.work_dir,"depthsplat")
            
            if cfg.validation_vis_progress:
                os.makedirs(GT_vis_output_folder,exist_ok=True)
                os.makedirs(depthsplat_rendered_output_folder,exist_ok=True)
            
            
            for i_iter_val, batch_val in enumerate(val_dataloader):
                print("Processed {}/{}".format(i_iter_val,len(val_dataloader)))
                # overall_val_batch_save_dir = osp.join(cfg.output_dir, cfg.exp_name, "validation")
                # os.makedirs(overall_val_batch_save_dir,exist_ok=True)
                # val_batch_save_dir = os.path.join(overall_val_batch_save_dir,"batch-{}".format(i_iter_val))

                bin_token = batch_val['bin_token'][0]
                
                

                
                if args.gpus<=1:
                    # forward here 
                    # get the psnr, ssim, mae and mse as well as the saved the visualization results
                    output_rgb_meter_dict,output_depth_meter_dict,input_depth_meter_dict,rendered_images,rendered_depth,gt_images,gt_depths = my_model.validation_step_with_token_names(batch_val, None,cfg)
                else:
                    output_rgb_meter_dict,output_depth_meter_dict,input_depth_meter_dict,rendered_images,rendered_depth,gt_images,gt_depths = my_model.module.validation_step_with_token_names(batch_val, None,cfg)
            
                if cfg.validation_vis_progress:
                                    # Omni-Scene
                    plt.figure(figsize=(20,10))
                    plt.subplot(3,2,1)
                    plt.axis('off')
                    plt.title("F-(l):Input")
                    plt.imshow(rendered_images[0].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,2)
                    plt.axis('off')
                    plt.title("F-(r):Input")
                    plt.imshow(rendered_images[1].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,3)
                    plt.axis('off')
                    plt.title("C-(l)")
                    plt.imshow(rendered_images[2].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,4)
                    plt.axis('off')
                    plt.title("C-(r)")
                    plt.imshow(rendered_images[3].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,5)
                    plt.axis('off')
                    plt.title("L-(l)")
                    plt.imshow(rendered_images[4].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,6)
                    plt.axis('off')
                    plt.title("L-(r)")
                    plt.imshow(rendered_images[5].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.savefig(os.path.join(depthsplat_rendered_output_folder,
                                bin_token+"_rendered_RGB.png"),bbox_inches='tight')
                    
                    # Omni-Scene Disparity
                    plt.figure(figsize=(20,10))
                    plt.subplot(3,2,1)
                    plt.axis('off')
                    plt.title("F-(l):Input")
                    plt.imshow(convert_depth_to_disp(depth=rendered_depth[0].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,2)
                    plt.axis('off')
                    plt.title("F-(r):Input")
                    plt.imshow(convert_depth_to_disp(depth=rendered_depth[1].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,3)
                    plt.axis('off')
                    plt.title("C-(l)")
                    plt.imshow(convert_depth_to_disp(depth=rendered_depth[2].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,4)
                    plt.axis('off')
                    plt.title("C-(r)")
                    plt.imshow(convert_depth_to_disp(depth=rendered_depth[3].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,5)
                    plt.axis('off')
                    plt.title("L-(l)")
                    plt.imshow(convert_depth_to_disp(depth=rendered_depth[4].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,6)
                    plt.axis('off')
                    plt.title("L-(r)")
                    plt.imshow(convert_depth_to_disp(depth=rendered_depth[5].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.savefig(os.path.join(depthsplat_rendered_output_folder,
                                bin_token+"_rendered_Depth.png"),bbox_inches='tight')


                
                    # RGB
                    plt.figure(figsize=(20,10))
                    plt.subplot(3,2,1)
                    plt.axis('off')
                    plt.title("F-(l):INPUT")
                    plt.imshow(gt_images[0].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,2)
                    plt.axis('off')
                    plt.title("F-(r):INPUT")
                    plt.imshow(gt_images[1].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,3)
                    plt.axis('off')
                    plt.title("C-(l)")
                    plt.imshow(gt_images[2].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,4)
                    plt.axis('off')
                    plt.title("C-(r)")
                    plt.imshow(gt_images[3].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,5)
                    plt.axis('off')
                    plt.title("L-(l)")
                    plt.imshow(gt_images[4].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.subplot(3,2,6)
                    plt.axis('off')
                    plt.title("L-(r)")
                    plt.imshow(gt_images[5].squeeze(0).permute(1,2,0).cpu().numpy())
                    plt.savefig(os.path.join(GT_vis_output_folder,
                                bin_token+"GT_RGB.png"),bbox_inches='tight')
                    
                    # Voxel Disparity
                    plt.figure(figsize=(20,10))
                    plt.subplot(3,2,1)
                    plt.axis('off')
                    plt.title("F-(l):Input")
                    plt.imshow(convert_depth_to_disp(depth=gt_depths[0].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,2)
                    plt.axis('off')
                    plt.title("F-(r):Input")
                    plt.imshow(convert_depth_to_disp(depth=gt_depths[1].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,3)
                    plt.axis('off')
                    plt.title("C-(l)")
                    plt.imshow(convert_depth_to_disp(depth=gt_depths[2].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,4)
                    plt.axis('off')
                    plt.title("C-(r)")
                    plt.imshow(convert_depth_to_disp(depth=gt_depths[3].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,5)
                    plt.axis('off')
                    plt.title("L-(l)")
                    plt.imshow(convert_depth_to_disp(depth=gt_depths[4].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.subplot(3,2,6)
                    plt.axis('off')
                    plt.title("L-(r)")
                    plt.imshow(convert_depth_to_disp(depth=gt_depths[5].squeeze(0).squeeze(0).cpu().numpy()))
                    plt.savefig(os.path.join(GT_vis_output_folder,
                                bin_token+"GT_Depth.png"),bbox_inches='tight')

            
            
            
                # saved the intermeidate results here for the RGB Here
                output_rgb_meter_center_view_left_psnr = output_rgb_meter_dict['center_view']['left']['psnr']
                output_rgb_meter_center_view_left_ssim = output_rgb_meter_dict['center_view']['left']['ssim']
                output_rgb_meter_center_view_right_psnr = output_rgb_meter_dict['center_view']['right']['psnr']
                output_rgb_meter_center_view_right_ssim = output_rgb_meter_dict['center_view']['right']['ssim']   
                output_rgb_meter_center['left'].update(output_rgb_meter_center_view_left_psnr,output_rgb_meter_center_view_left_ssim)
                output_rgb_meter_center['right'].update(output_rgb_meter_center_view_right_psnr,output_rgb_meter_center_view_right_ssim)
                
                output_rgb_meter_first_view_left_psnr = output_rgb_meter_dict['first_view']['left']['psnr']
                output_rgb_meter_first_view_left_ssim = output_rgb_meter_dict['first_view']['left']['ssim']   
                output_rgb_meter_first_view_right_psnr = output_rgb_meter_dict['first_view']['right']['psnr']
                output_rgb_meter_first_view_right_ssim = output_rgb_meter_dict['first_view']['right']['ssim']
                output_rgb_meter_first['left'].update(output_rgb_meter_first_view_left_psnr,output_rgb_meter_first_view_left_ssim)
                output_rgb_meter_first['right'].update(output_rgb_meter_first_view_right_psnr,output_rgb_meter_first_view_right_ssim)
                
                output_rgb_meter_last_view_left_psnr = output_rgb_meter_dict['last_view']['left']['psnr']
                output_rgb_meter_last_view_left_ssim = output_rgb_meter_dict['last_view']['left']['ssim']   
                output_rgb_meter_last_view_right_psnr = output_rgb_meter_dict['last_view']['right']['psnr']
                output_rgb_meter_last_view_right_ssim = output_rgb_meter_dict['last_view']['right']['ssim']
                output_rgb_meter_last["left"].update(output_rgb_meter_last_view_left_psnr,output_rgb_meter_last_view_left_ssim)
                output_rgb_meter_last["right"].update(output_rgb_meter_last_view_right_psnr,output_rgb_meter_last_view_right_ssim)
                
                
                # saved the intermeidate results here for the Depth here
                output_depth_meter_center_view_left_mae = output_depth_meter_dict['center_view']['left']['mae']
                output_depth_meter_center_view_left_mse = output_depth_meter_dict['center_view']['left']['mse']
                output_depth_meter_center_view_right_mae = output_depth_meter_dict['center_view']['right']['mae']
                output_depth_meter_center_view_right_mse = output_depth_meter_dict['center_view']['right']['mse']
                output_depth_meter_center["left"].update(mae=output_depth_meter_center_view_left_mae,
                                                        mse=output_depth_meter_center_view_left_mse)
                output_depth_meter_center["right"].update(mae=output_depth_meter_center_view_right_mae,
                                                        mse=output_depth_meter_center_view_right_mse)

                output_depth_meter_first_view_left_mae = output_depth_meter_dict['first_view']['left']['mae']
                output_depth_meter_first_view_left_mse = output_depth_meter_dict['first_view']['left']['mse']
                output_depth_meter_first_view_right_mae = output_depth_meter_dict['first_view']['right']['mae']
                output_depth_meter_first_view_right_mse = output_depth_meter_dict['first_view']['right']['mse']
                output_depth_meter_first["left"].update(mae=output_depth_meter_first_view_left_mae,
                                                        mse=output_depth_meter_first_view_left_mse)
                output_depth_meter_first["right"].update(mae=output_depth_meter_first_view_right_mae,
                                                        mse=output_depth_meter_first_view_right_mse)
                

                output_depth_meter_last_view_left_mae = output_depth_meter_dict['last_view']['left']['mae']
                output_depth_meter_last_view_left_mse = output_depth_meter_dict['last_view']['left']['mse']
                output_depth_meter_last_view_right_mae = output_depth_meter_dict['last_view']['right']['mae']
                output_depth_meter_last_view_right_mse = output_depth_meter_dict['last_view']['right']['mse']
                output_depth_meter_last["left"].update(mae=output_depth_meter_last_view_left_mae,
                                                        mse=output_depth_meter_last_view_left_mse)
                output_depth_meter_last["right"].update(mae=output_depth_meter_last_view_right_mae,
                                                        mse=output_depth_meter_last_view_right_mse)
                
                # saved the input results
                
                input_depth_meter_left_mae = input_depth_meter_dict['input_depth']['left']['mae']
                input_depth_meter_left_mse = input_depth_meter_dict['input_depth']['left']['mse']
                input_depth_meter_right_mae = input_depth_meter_dict['input_depth']['right']['mae']
                input_depth_meter_right_mse = input_depth_meter_dict['input_depth']['right']['mse']       

                input_depth_meter['left'].update(mae=input_depth_meter_left_mae,
                                                    mse=input_depth_meter_left_mse)
                input_depth_meter['right'].update(mae=input_depth_meter_right_mae,
                                                    mse=input_depth_meter_right_mse)
                
            
            # for this
            output_rgb_meter_center['left'] = output_rgb_meter_center['left'].get_stats()
            output_rgb_meter_center['right'] = output_rgb_meter_center['right'].get_stats()
            
            output_rgb_meter_first['left'] = output_rgb_meter_first['left'].get_stats()
            output_rgb_meter_first['right'] = output_rgb_meter_first['right'].get_stats()
            
            output_rgb_meter_last['left'] = output_rgb_meter_last['left'].get_stats()
            output_rgb_meter_last['right'] = output_rgb_meter_last['right'].get_stats()
            
            output_depth_meter_center['left'] = output_depth_meter_center['left'].get_stats()
            output_depth_meter_center['right'] = output_depth_meter_center['right'].get_stats()

            output_depth_meter_first['left'] = output_depth_meter_first['left'].get_stats()
            output_depth_meter_first['right'] = output_depth_meter_first['right'].get_stats()
            
            output_depth_meter_last['left'] = output_depth_meter_last['left'].get_stats()
            output_depth_meter_last['right'] = output_depth_meter_last['right'].get_stats()
            
            
            input_depth_meter['left'] = input_depth_meter['left'].get_stats()
            input_depth_meter['right'] = input_depth_meter['right'].get_stats()
            
            
            results_dict = {
            "rgb_center": output_rgb_meter_center,
            "rgb_first": output_rgb_meter_first,
            "rgb_last": output_rgb_meter_last,
            "depth_center":output_depth_meter_center,
            "depth_first":output_depth_meter_first,
            "depth_last":output_depth_meter_last,
            "input_depth":input_depth_meter
                
            }
            
            if not cfg.validation_vis_progress:
                saved_into_json(data_dict=results_dict,
                                path=os.path.join(cfg.work_dir,"metric.json"))







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
