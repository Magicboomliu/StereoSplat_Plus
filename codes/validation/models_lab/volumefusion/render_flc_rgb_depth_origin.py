import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from diffusers.optimization import get_scheduler
import math
import sys
sys.path.append("..")
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
import numpy as np
from torch import Tensor,nn
from tools.metrics import RGB_Quality_Meter,Depth_Quality_Meter,saved_into_json
from mmengine.registry import MODELS

# define the models
from models_lab.VolumeFusion.volumefusion import VolumeFusion

import matplotlib.pyplot as plt


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

def convert_depth_to_disp(factor=328.318735,depth=None):
    
    mask = depth>0
    mask = mask.astype(np.float32)

    disparity = factor / (depth +1e-3)
    disparity = disparity * mask
    disparity = np.clip(disparity,a_max=220,a_min=0)
    
    disparity = kitti_colormap(disparity)
    return disparity

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

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)
    
    #################### Dataset Configurration #############################
    dataset_config = cfg.dataset_params

    # generate datasets
    dataset = getattr(datasets, dataset_config.dataset_name)

    dataset_config.val_filelist = args.val_filelist

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
    val_dataset = dataset(**val_params)
    
    val_dataloader = DataLoader(
        val_dataset, dataset_config.batch_size_val, shuffle=False,
        num_workers=dataset_config.num_workers_val
    )


    # Define the Model/Optimizer/Schduler Here
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
    

    # move to the accelerate
    my_model, val_dataloader = accelerator.prepare(
        my_model, val_dataloader
    )
    
    model_path = args.resume_from
    if model_path =="none":
        model_path =None
    if model_path is not None:
        accelerator.load_state(model_path, map_location='cpu', strict=False)

    if accelerator.is_main_process:
        print('successfully resumed from {}'.format(model_path))


    # replace here
    cfg.output_dir = args.work_dir
    os.makedirs(cfg.output_dir,exist_ok=True)

    overall_val_batch_save_dir = cfg.output_dir
    
    GT_Folder = os.path.join(overall_val_batch_save_dir,"GT")
    os.makedirs(GT_Folder,exist_ok=True)

    
    # do the visualization here
    with torch.no_grad():
        my_model.eval()


        if accelerator.is_main_process:
            
            
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
                
                bin_token = batch_val['bin_token'][0]

                if args.gpus<=1:
                    # forward here 
                    # get the psnr, ssim, mae and mse as well as the saved the visualization results
                   metrics_rendered_rgb_list,metrics_rendered_depth_list,metrics_estimated_depth_list,final_rendered_rgb_list,final_rendered_depth_list,final_gt_rgb_list,final_gt_depth_list = my_model.validation_step_with_token_names(batch_val, None,cfg)
                else:
                    metrics_rendered_rgb_list,metrics_rendered_depth_list,metrics_estimated_depth_list,final_rendered_rgb_list,final_rendered_depth_list,final_gt_rgb_list,final_gt_depth_list = my_model.module.validation_step_with_token_names(batch_val, None,cfg)

                for i in range(len(metrics_rendered_rgb_list)):
                    
                    # doing the visualization here
                    rendered_rgb_images_data = final_rendered_rgb_list[i].squeeze(0)
                    rendered_depth_data = final_rendered_depth_list[i].squeeze(0)
                    gt_rgb_images_data = final_gt_rgb_list[i].squeeze(0)
                    gt_depth_data = final_gt_depth_list[i].squeeze(0)
                    
     
                    
                    if i==0:
                        saved_folder = os.path.join(overall_val_batch_save_dir,"fusion")
                    elif i ==1:
                        saved_folder = os.path.join(overall_val_batch_save_dir,"volume")
                    elif i==2:
                        saved_folder = os.path.join(overall_val_batch_save_dir,"cv")
                    
                    os.makedirs(saved_folder,exist_ok=True)
                    
                    if args.output_vis:
                        plt.figure(figsize=(20,10))
                        plt.subplot(3,2,1)
                        plt.axis('off')
                        plt.title("F-(l):Input")
                        plt.imshow(rendered_rgb_images_data[4].permute(1,2,0).cpu().numpy())
                        plt.subplot(3,2,2)
                        plt.axis('off')
                        plt.title("F-(r):Input")
                        plt.imshow(rendered_rgb_images_data[5].permute(1,2,0).cpu().numpy())
                        plt.subplot(3,2,3)
                        plt.axis('off')
                        plt.title("C-(l)")
                        plt.imshow(rendered_rgb_images_data[0].permute(1,2,0).cpu().numpy())
                        plt.subplot(3,2,4)
                        plt.axis('off')
                        plt.title("C-(r)")
                        plt.imshow(rendered_rgb_images_data[2].permute(1,2,0).cpu().numpy())
                        plt.subplot(3,2,5)
                        plt.axis('off')
                        plt.title("L-(l)")
                        plt.imshow(rendered_rgb_images_data[1].permute(1,2,0).cpu().numpy())
                        plt.subplot(3,2,6)
                        plt.axis('off')
                        plt.title("L-(r)")
                        plt.imshow(rendered_rgb_images_data[3].permute(1,2,0).cpu().numpy())
                        plt.savefig(os.path.join(saved_folder,
                                    bin_token+"_rendered_RGB.png"),bbox_inches='tight')
                        
                        # Omni-Scene Disparity
                        plt.figure(figsize=(20,10))
                        plt.subplot(3,2,1)
                        plt.axis('off')
                        plt.title("F-(l):Input")
                        plt.imshow(convert_depth_to_disp(depth=rendered_depth_data[4].cpu().numpy()))
                        plt.subplot(3,2,2)
                        plt.axis('off')
                        plt.title("F-(r):Input")
                        plt.imshow(convert_depth_to_disp(depth=rendered_depth_data[5].cpu().numpy()))
                        plt.subplot(3,2,3)
                        plt.axis('off')
                        plt.title("C-(l)")
                        plt.imshow(convert_depth_to_disp(depth=rendered_depth_data[0].cpu().numpy()))
                        plt.subplot(3,2,4)
                        plt.axis('off')
                        plt.title("C-(r)")
                        plt.imshow(convert_depth_to_disp(depth=rendered_depth_data[2].cpu().numpy()))
                        plt.subplot(3,2,5)
                        plt.axis('off')
                        plt.title("L-(l)")
                        plt.imshow(convert_depth_to_disp(depth=rendered_depth_data[1].cpu().numpy()))
                        plt.subplot(3,2,6)
                        plt.axis('off')
                        plt.title("L-(r)")
                        plt.imshow(convert_depth_to_disp(depth=rendered_depth_data[3].cpu().numpy()))
                        plt.savefig(os.path.join(saved_folder,
                                    bin_token+"_rendered_Depth.png"),bbox_inches='tight')


                    
                        # RGB
                        plt.figure(figsize=(20,10))
                        plt.subplot(3,2,1)
                        plt.axis('off')
                        plt.title("F-(l):INPUT")
                        plt.imshow(gt_rgb_images_data[4].permute(1,2,0).cpu().numpy())
                        plt.subplot(3,2,2)
                        plt.axis('off')
                        plt.title("F-(r):INPUT")
                        plt.imshow(gt_rgb_images_data[5].permute(1,2,0).cpu().numpy())
                        plt.subplot(3,2,3)
                        plt.axis('off')
                        plt.title("C-(l)")
                        plt.imshow(gt_rgb_images_data[0].permute(1,2,0).cpu().numpy())
                        plt.subplot(3,2,4)
                        plt.axis('off')
                        plt.title("C-(r)")
                        plt.imshow(gt_rgb_images_data[2].permute(1,2,0).cpu().numpy())
                        plt.subplot(3,2,5)
                        plt.axis('off')
                        plt.title("L-(l)")
                        plt.imshow(gt_rgb_images_data[1].permute(1,2,0).cpu().numpy())
                        plt.subplot(3,2,6)
                        plt.axis('off')
                        plt.title("L-(r)")
                        plt.imshow(gt_rgb_images_data[3].permute(1,2,0).cpu().numpy())
                        plt.savefig(os.path.join(GT_Folder,
                                    bin_token+"GT_RGB.png"),bbox_inches='tight')
                        
                        # Voxel Disparity
                        plt.figure(figsize=(20,10))
                        plt.subplot(3,2,1)
                        plt.axis('off')
                        plt.title("F-(l):Input")
                        plt.imshow(convert_depth_to_disp(depth=gt_depth_data[4].cpu().numpy()))
                        plt.subplot(3,2,2)
                        plt.axis('off')
                        plt.title("F-(r):Input")
                        plt.imshow(convert_depth_to_disp(depth=gt_depth_data[5].cpu().numpy()))
                        plt.subplot(3,2,3)
                        plt.axis('off')
                        plt.title("C-(l)")
                        plt.imshow(convert_depth_to_disp(depth=gt_depth_data[0].cpu().numpy()))
                        plt.subplot(3,2,4)
                        plt.axis('off')
                        plt.title("C-(r)")
                        plt.imshow(convert_depth_to_disp(depth=gt_depth_data[2].cpu().numpy()))
                        plt.subplot(3,2,5)
                        plt.axis('off')
                        plt.title("L-(l)")
                        plt.imshow(convert_depth_to_disp(depth=gt_depth_data[1].cpu().numpy()))
                        plt.subplot(3,2,6)
                        plt.axis('off')
                        plt.title("L-(r)")
                        plt.imshow(convert_depth_to_disp(depth=gt_depth_data[3].cpu().numpy()))
                        plt.savefig(os.path.join(GT_Folder,
                                    bin_token+"GT_Depth.png"),bbox_inches='tight')
                    
                    
                    

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
            
            if not args.output_vis:
                saved_into_json(data_dict=results_dict_fusion,
                                path=os.path.join(overall_val_batch_save_dir,"fusion_metric.json"))

                saved_into_json(data_dict=results_dict_volume,
                                path=os.path.join(overall_val_batch_save_dir,"volume_metric.json"))

                saved_into_json(data_dict=results_dict_cv,
                                path=os.path.join(overall_val_batch_save_dir,"cv_metric.json"))





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
