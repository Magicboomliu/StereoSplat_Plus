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
from models_lab.VolumeFusion.volumefusion import VolumeFusion
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
from evaluations.data_container.simple_datareader import get_inputs_info,Get_First_Key_Frame_LiDAR_To_World
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import cv2
from model.utils.image import resize_image,HWC3
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


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

def get_diff_elements(A, B):
    """
    返回 B 中存在但 A 中没有的元素

    参数:
        A (list): 较小的子列表
        B (list): 包含 A 的完整列表

    返回:
        list: B 中不在 A 中的元素
    """
    return [item for item in B if item not in A]

def load_pkl(filepath):

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

class FusionConfigurationDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: List of data_dict, 每个 data_dict 是你上面的结构
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def main(args):
    cfg = Config.fromfile(args.config_path)
    cfg.work_dir = args.output_folder


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
    
    dataset_config.semi_global_folder_path = args.semi_global_map

    val_params = {
        "datapath":dataset_config.datapath,
        "map_path":dataset_config.semi_global_folder_path,
        "resolution":dataset_config.resolution, 
        "split":"val",
        "sequence":dataset_config.sequence,
        "depth_info_dict":dataset_config.depth_info_params,
        "camera_model": dataset_config.camera_model,
    }


    map_names_list = os.listdir(dataset_config.semi_global_folder_path)
    map_names_list.remove("global.pkl")
    semi_global_map_list = sorted(map_names_list)
    
    
    # saved output folder path
    output_saved_folder_path = os.path.join(args.output_folder,
                                            args.ablation_type)
    os.makedirs(output_saved_folder_path,exist_ok=True)

    # Define the Model/Optimizer/Schduler Here
    my_model = VolumeFusion(backbone=cfg.model.backbone,
                            neck=cfg.model.neck,
                            costvolume_gs=cfg.model.costvolume_gs,
                            volume_gs=cfg.model.volume_gs,
                            losses_params=cfg.model.losses_params,
                            camera_args=cfg.camera_args,
                            dataset_params=cfg.dataset_params,
                            use_checkpoint=cfg.use_checkpoint)


    my_model= accelerator.prepare(
        my_model
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
    
    volumefusion_renderer = my_model.renderer
    
    for idx, semi_global_info_path in enumerate(semi_global_map_list):
        
        semi_global_info_path = os.path.join(dataset_config.semi_global_folder_path,semi_global_info_path)
        semi_global_info = load_pkl(semi_global_info_path)
        
        print("Processing the Semi-Global-Map ID {}/{}".format(idx,len(semi_global_map_list)))
        key_input_frames_idx, all_frames_idx = semi_global_info['key_frames_list'], semi_global_info['all_frames_list']

        # get the first key input frame's LiDAR as the world orgin: where is 4x4
        first_key_frame_lidar_to_world_pose = Get_First_Key_Frame_LiDAR_To_World(val_params['datapath'],
                                                                                 key_input_frames_idx[0].replace("annotations","annotations_simple"))
        
        # using the fusion GS for Rendering.
        validation_views = get_diff_elements(A=key_input_frames_idx,B=all_frames_idx)
        
        # Start Fusion Here
        input_key_frame_info_list = []
        
        rendered_images_all_list = []
        rendered_depths_all_list = []

        saved_rendered_image_video_path = os.path.join(output_saved_folder_path,"/".join(key_input_frames_idx[0].split("/")[:2]).replace("annotations","rendered_rgb_videos"),"rendered_videos_semi_map_{}.mp4".format(idx))
        os.makedirs(os.path.dirname(saved_rendered_image_video_path),exist_ok=True)
                

        saved_rendered_depth_video_path = os.path.join(output_saved_folder_path,"/".join(key_input_frames_idx[0].split("/")[:2]).replace("annotations","rendered_depth_videos"),"rendered_videos_semi_map_{}.mp4".format(idx))
        os.makedirs(os.path.dirname(saved_rendered_depth_video_path),exist_ok=True)
        
        
        saved_rendered_results_json_path = os.path.join(output_saved_folder_path,"/".join(key_input_frames_idx[0].split("/")[:2]).replace("annotations","rendered_quality_json"),"rendered_quality_semi_map_{}.json".format(idx))
        os.makedirs(os.path.dirname(saved_rendered_results_json_path),exist_ok=True)

        
        print("Step 1:  building semi-global map dataloader for fusion...")
        for input_frame_name in tqdm(key_input_frames_idx):
            
            input_annotation_name = input_frame_name.replace("annotations","annotations_simple")
            # current_input_infos
            input_infos_list = get_inputs_info(datapath=val_params['datapath'],
                        reso = val_params['resolution'],
                        first_ref=first_key_frame_lidar_to_world_pose,
                        simple_annotation_path_list=[input_annotation_name],
                        depth_info_params =val_params['depth_info_dict'],
                        extra_list=[])
            
            input_key_frame_info = input_infos_list[0]
            
            input_key_frame_info_list.append(input_key_frame_info)

        configuration_fuse_dataset = FusionConfigurationDataset(input_key_frame_info_list)
        configuration_fuse_dataloader = DataLoader(configuration_fuse_dataset, batch_size=1, shuffle=False)
        configuration_fuse_dataloader = accelerator.prepare(
                                        configuration_fuse_dataloader)
        
        # For the Evaluation Metrics
        
        renderd_left_image_metrics = Basic_Meter(psnr=0,ssim=0,mae=0,mse=0)
        rendered_right_image_metrics = Basic_Meter(psnr=0,ssim=0,mae=0,mse=0)
        
        
        rendered_index = 0
        print("Step 2: begin incremenatl gaussain fusion.....")
    
        if args.ablation_type=='simple_fusion':
            global_gaussains_list = []
            
            with torch.no_grad():
                my_model.eval()
                
                key_frame_index = 0
                
                for batch in tqdm(configuration_fuse_dataloader):    
                    ego_lidar_gaussain = my_model.filewise_inference_only(batch,cfg=cfg)
                    ego_to_world_pose = batch['input']['lidar_to_world'][0][0]
                    ego_lidar_gaussain_shifted = transform_gs_to_given_pose(g2=ego_lidar_gaussain,
                                            c2w=ego_to_world_pose)
                    
                    
                    # if key_frame_index<10:
                    #     torch.save(ego_lidar_gaussain_shifted, "key_frame_gs_{}.pt".format(key_frame_index)) 
                    
                    key_frame_index = key_frame_index + 1
                
                                   
                    global_gaussains_list.append(ego_lidar_gaussain_shifted)
     
            global_gaussains = torch.cat(global_gaussains_list,dim=1)

            print("Step 3:  Building the Inference Dataset...")
            all_key_frame_info_list = []
            for input_frame_name in tqdm(all_frames_idx):
                input_annotation_name = input_frame_name.replace("annotations","annotations_simple")
            
                # current_input_infos
                input_infos_list = get_inputs_info(datapath=val_params['datapath'],
                            reso = val_params['resolution'],
                            first_ref=first_key_frame_lidar_to_world_pose,
                            simple_annotation_path_list=[input_annotation_name],
                            depth_info_params =val_params['depth_info_dict'],
                            extra_list=[])
                
                input_key_frame_info = input_infos_list[0]
                
                all_key_frame_info_list.append(input_key_frame_info)

            configuration_fuse_dataset_for_eval = FusionConfigurationDataset(all_key_frame_info_list)
            configuration_fuse_dataloader_for_eval = DataLoader(configuration_fuse_dataset_for_eval, batch_size=1, shuffle=False)
            configuration_fuse_dataloader_for_eval = accelerator.prepare(
                                            configuration_fuse_dataloader_for_eval)

            # Rendered Images and the Evaluations
            rendered_index = 0
            volumefusion_renderer = my_model.renderer
            for batch in tqdm(configuration_fuse_dataloader_for_eval):
                
                current_annotation_path = all_frames_idx[rendered_index]
                
                saved_rendered_image_name = current_annotation_path.replace("annotations","rendered_images").replace(".json",".png")
                saved_rendered_image_name = os.path.join(output_saved_folder_path,saved_rendered_image_name)
                os.makedirs(os.path.dirname(saved_rendered_image_name),exist_ok=True)

                saved_rendered_depth_name = current_annotation_path.replace("annotations","rendered_depth").replace(".json",".png")
                saved_rendered_depth_name = os.path.join(output_saved_folder_path,saved_rendered_depth_name)
                os.makedirs(os.path.dirname(saved_rendered_depth_name),exist_ok=True)
                

                rendered_c2w = batch["output"]["c2w"].to(accelerator.device)
                fovxs = batch["output"]["fovxs"].to(accelerator.device)
                fovys = batch["output"]["fovys"].to(accelerator.device)
                

                rendered_results =volumefusion_renderer.render(
                    gaussians=global_gaussains,
                    c2w=rendered_c2w,
                    fovx=fovxs,
                    fovy=fovys,
                    rays_o=None,
                    rays_d=None
                )
                
                rendered_image = rendered_results['image'] #(B,V,3,H,W)
                rendered_depth = rendered_results['depth'] #(B,V,1,H,W)

                gt_images = batch['output']['imgs'].to(accelerator.device)
                gt_sparse_depth = batch['output']["sparse_gts"].to(accelerator.device).unsqueeze(2)
                

                    
                current_left_psnr, current_left_ssim = compute_psnr_ssim(pred=rendered_image[:,0,:,:,:],
                                                                         target=gt_images[:,0,:,:,:]
                                                                         )
                current_right_psnr, current_right_ssim = compute_psnr_ssim(pred=rendered_image[:,1,:,:,:],
                                                                         target=gt_images[:,1,:,:,:]
                                                                         )
                
                current_left_mae, current_left_mse = compute_depth_mae_mse(depth_pred=rendered_depth[:,0,:,:,:],
                                                                           depth_gt=gt_sparse_depth[:,0,:,:,:])

                current_right_mae, current_right_mse = compute_depth_mae_mse(depth_pred=rendered_depth[:,1,:,:,:],
                                                                           depth_gt=gt_sparse_depth[:,1,:,:,:])
                
                
                renderd_left_image_metrics.update(psnr=current_left_psnr.data.item(),
                                                  ssim=current_left_ssim.data.item(),
                                                  mae=current_left_mae.data.item(),
                                                  mse=current_left_mse.data.item())
                
                rendered_right_image_metrics.update(psnr=current_right_psnr.data.item(),
                                                  ssim=current_right_ssim.data.item(),
                                                  mae=current_right_mae.data.item(),
                                                  mse=current_right_mse.data.item()
                )
                

                if args.output_vis:
                    
                    # saved rendered images
                    rendered_image_left = rendered_image[0][0].permute(1,2,0) #(H,W,3)
                    rendered_image_right = rendered_image[0][1].permute(1,2,0) #(H,W,3)
                    rendered_image_for_vis = torch.cat((rendered_image_left,rendered_image_right),dim=1).cpu().numpy()
                    
                    skimage.io.imsave(saved_rendered_image_name,(rendered_image_for_vis*255).astype(np.uint8))

                    # saved rendered depths
                    rendered_depth_left = rendered_depth[0][0][0]
                    rendered_depth_right = rendered_depth[0][1][0]
                    rendered_depth_for_vis = torch.cat((rendered_depth_left,rendered_depth_right),dim=1).cpu().numpy()
                    rendered_depth_for_vis = clean_and_clip(rendered_depth_for_vis)
                    rendered_depth_for_vis = convert_depth_to_disp(factor=328.318735,depth=rendered_depth_for_vis)
                    skimage.io.imsave(saved_rendered_depth_name,rendered_depth_for_vis)
                    
                    rendered_images_all_list.append((rendered_image_for_vis*255).astype(np.uint8))
                    rendered_depths_all_list.append(rendered_depth_for_vis)
                    
                rendered_index = rendered_index + 1
                
        elif args.ablation_type=='no_fusion':
            print("building the inference dataset........")
            all_key_frame_info_list = []
            for input_frame_name in tqdm(all_frames_idx):
                input_annotation_name = input_frame_name.replace("annotations","annotations_simple")
            
                # current_input_infos
                input_infos_list = get_inputs_info(datapath=val_params['datapath'],
                            reso = val_params['resolution'],
                            first_ref=first_key_frame_lidar_to_world_pose,
                            simple_annotation_path_list=[input_annotation_name],
                            depth_info_params =val_params['depth_info_dict'],
                            extra_list=[])
                
                input_key_frame_info = input_infos_list[0]
                
                all_key_frame_info_list.append(input_key_frame_info)

            configuration_fuse_dataset_for_eval = FusionConfigurationDataset(all_key_frame_info_list)
            configuration_fuse_dataloader_for_eval = DataLoader(configuration_fuse_dataset_for_eval, batch_size=1, shuffle=False)
            configuration_fuse_dataloader_for_eval = accelerator.prepare(
                                            configuration_fuse_dataloader_for_eval)
            
            
            rendered_index = 0
            volumefusion_renderer = my_model.renderer
            my_model.eval()
            current_global_gs_stack = []
            for batch in tqdm(configuration_fuse_dataloader_for_eval):
                
                
                current_annotation_path = all_frames_idx[rendered_index]
                saved_rendered_image_name = current_annotation_path.replace("annotations","rendered_images").replace(".json",".png")
                saved_rendered_image_name = os.path.join(output_saved_folder_path,saved_rendered_image_name)
                os.makedirs(os.path.dirname(saved_rendered_image_name),exist_ok=True)

                saved_rendered_depth_name = current_annotation_path.replace("annotations","rendered_depth").replace(".json",".png")
                saved_rendered_depth_name = os.path.join(output_saved_folder_path,saved_rendered_depth_name)
                os.makedirs(os.path.dirname(saved_rendered_depth_name),exist_ok=True)

            
                # if key_frame: updata current GS
                if current_annotation_path in key_input_frames_idx:
                    with torch.no_grad():
                        ego_lidar_gaussain = my_model.filewise_inference_only(batch,cfg=cfg)
                        ego_to_world_pose = batch['input']['lidar_to_world'][0][0]
                        ego_lidar_gaussain_shifted = transform_gs_to_given_pose(g2=ego_lidar_gaussain,
                                                c2w=ego_to_world_pose)
                        
                        # FIXME
                        if rendered_index==0:
                            current_global_gs_stack.append(ego_lidar_gaussain_shifted)
                        else:
                            assert len(current_global_gs_stack)==1
                            current_global_gs_stack.pop()
                            current_global_gs_stack.append(ego_lidar_gaussain_shifted)
                
                # rendered here

                rendered_c2w = batch["output"]["c2w"].to(accelerator.device)
                fovxs = batch["output"]["fovxs"].to(accelerator.device)
                fovys = batch["output"]["fovys"].to(accelerator.device)
                
                assert len(current_global_gs_stack)==1
                rendered_results =volumefusion_renderer.render(
                    gaussians=current_global_gs_stack[0],
                    c2w=rendered_c2w,
                    fovx=fovxs,
                    fovy=fovys,
                    rays_o=None,
                    rays_d=None
                )
                
                rendered_image = rendered_results['image'] #(B,V,3,H,W)
                rendered_depth = rendered_results['depth'] #(B,V,1,H,W)

                gt_images = batch['output']['imgs'].to(accelerator.device)
                gt_sparse_depth = batch['output']["sparse_gts"].to(accelerator.device).unsqueeze(2)
                    
                current_left_psnr, current_left_ssim = compute_psnr_ssim(pred=rendered_image[:,0,:,:,:],
                                                                         target=gt_images[:,0,:,:,:]
                                                                         )
                current_right_psnr, current_right_ssim = compute_psnr_ssim(pred=rendered_image[:,1,:,:,:],
                                                                         target=gt_images[:,1,:,:,:]
                                                                         )
                
                current_left_mae, current_left_mse = compute_depth_mae_mse(depth_pred=rendered_depth[:,0,:,:,:],
                                                                           depth_gt=gt_sparse_depth[:,0,:,:,:])

                current_right_mae, current_right_mse = compute_depth_mae_mse(depth_pred=rendered_depth[:,1,:,:,:],
                                                                           depth_gt=gt_sparse_depth[:,1,:,:,:])
                
                
                renderd_left_image_metrics.update(psnr=current_left_psnr.data.item(),
                                                  ssim=current_left_ssim.data.item(),
                                                  mae=current_left_mae.data.item(),
                                                  mse=current_left_mse.data.item())
                
                rendered_right_image_metrics.update(psnr=current_right_psnr.data.item(),
                                                  ssim=current_right_ssim.data.item(),
                                                  mae=current_right_mae.data.item(),
                                                  mse=current_right_mse.data.item()
                )


                if args.output_vis:
                    
                    # saved rendered images
                    rendered_image_left = rendered_image[0][0].permute(1,2,0) #(H,W,3)
                    rendered_image_right = rendered_image[0][1].permute(1,2,0) #(H,W,3)
                    rendered_image_for_vis = torch.cat((rendered_image_left,rendered_image_right),dim=1).cpu().numpy() 
                    skimage.io.imsave(saved_rendered_image_name,(rendered_image_for_vis*255).astype(np.uint8))

                    # saved rendered depths
                    rendered_depth_left = rendered_depth[0][0][0]
                    rendered_depth_right = rendered_depth[0][1][0]
                    rendered_depth_for_vis = torch.cat((rendered_depth_left,rendered_depth_right),dim=1).cpu().numpy()
                    rendered_depth_for_vis = clean_and_clip(rendered_depth_for_vis)
                    rendered_depth_for_vis = convert_depth_to_disp(factor=328.318735,depth=rendered_depth_for_vis)
                    skimage.io.imsave(saved_rendered_depth_name,rendered_depth_for_vis)
                    
                    rendered_images_all_list.append((rendered_image_for_vis*255).astype(np.uint8))
                    rendered_depths_all_list.append(rendered_depth_for_vis)
                
                
                
                rendered_index = rendered_index + 1

        elif args.ablation_type=='no_fusion_as_one_version':            
            with torch.no_grad():
                my_model.eval()
                incremental_frame_index = 0
                
                each_key_frame_gs_list = []
                
                for batch in tqdm(configuration_fuse_dataloader):    
                    ego_lidar_gaussain = my_model.filewise_inference_only(batch,cfg=cfg)
                    ego_to_world_pose = batch['input']['lidar_to_world'][0][0]
                    ego_lidar_gaussain_shifted = transform_gs_to_given_pose(g2=ego_lidar_gaussain,
                                            c2w=ego_to_world_pose)
                    
                    current_image_size = batch['input']['imgs'].shape[-2:]
                    current_keyframe_output_c2w = batch["output"]["c2w"].to(accelerator.device) #(B,V,4,4)
                    current_keyframe_output_ck = batch['input']['cks'].to(accelerator.device)   #(B,V,3,3)


                    if incremental_frame_index>0:
                        # remove the GS inside the global gaussains
                        global_gaussains = remove_gaussians_in_frustum(gaussians=global_gaussains[0],
                                                    c2w=current_keyframe_output_c2w[0],
                                                    intrinsics=current_keyframe_output_ck[0],
                                                    image_size=current_image_size)
                        # # # # FIXME

                        # kepted_3dgs_each = remove_gaussians_in_frustum(gaussians=each_key_frame_gs_list[incremental_frame_index-1][0],
                        #                             c2w=current_keyframe_output_c2w[0],
                        #                             intrinsics=current_keyframe_output_ck[0],
                        #                             image_size=[300,1200])
                        
                        
                        # if incremental_frame_index<11:
                            
                        #     torch.save(kepted_3dgs_each,"key_frame_gs_{}.pt".format(incremental_frame_index-1))
                        # else:
                        #     quit()
                        
                        
                        
                        global_gaussains = global_gaussains.unsqueeze(0)
                        # fusion
                        global_gaussains = torch.cat((global_gaussains,ego_lidar_gaussain_shifted),dim=1)
                    
                    else:
                        global_gaussains = ego_lidar_gaussain_shifted
                        
                    each_key_frame_gs_list.append(ego_lidar_gaussain_shifted)
                        
                    incremental_frame_index = incremental_frame_index + 1
                    

            print("Step 3:  Building the Inference Dataset...")
            all_key_frame_info_list = []
            for input_frame_name in tqdm(all_frames_idx):
                input_annotation_name = input_frame_name.replace("annotations","annotations_simple")
                # current_input_infos
                input_infos_list = get_inputs_info(datapath=val_params['datapath'],
                            reso = val_params['resolution'],
                            first_ref=first_key_frame_lidar_to_world_pose,
                            simple_annotation_path_list=[input_annotation_name],
                            depth_info_params =val_params['depth_info_dict'],
                            extra_list=[])
                
                input_key_frame_info = input_infos_list[0]
                
                all_key_frame_info_list.append(input_key_frame_info)

            configuration_fuse_dataset_for_eval = FusionConfigurationDataset(all_key_frame_info_list)
            configuration_fuse_dataloader_for_eval = DataLoader(configuration_fuse_dataset_for_eval, batch_size=1, shuffle=False)
            configuration_fuse_dataloader_for_eval = accelerator.prepare(
                                            configuration_fuse_dataloader_for_eval)

            # Rendered Images and the Evaluations
            rendered_index = 0
            volumefusion_renderer = my_model.renderer
            for batch in tqdm(configuration_fuse_dataloader_for_eval):
                
                current_annotation_path = all_frames_idx[rendered_index]
                
                saved_rendered_image_name = current_annotation_path.replace("annotations","rendered_images").replace(".json",".png")
                saved_rendered_image_name = os.path.join(output_saved_folder_path,saved_rendered_image_name)
                os.makedirs(os.path.dirname(saved_rendered_image_name),exist_ok=True)

                saved_rendered_depth_name = current_annotation_path.replace("annotations","rendered_depth").replace(".json",".png")
                saved_rendered_depth_name = os.path.join(output_saved_folder_path,saved_rendered_depth_name)
                os.makedirs(os.path.dirname(saved_rendered_depth_name),exist_ok=True)
                

                rendered_c2w = batch["output"]["c2w"].to(accelerator.device)
                fovxs = batch["output"]["fovxs"].to(accelerator.device)
                fovys = batch["output"]["fovys"].to(accelerator.device)
                
                rendered_results =volumefusion_renderer.render(
                    gaussians=global_gaussains,
                    c2w=rendered_c2w,
                    fovx=fovxs,
                    fovy=fovys,
                    rays_o=None,
                    rays_d=None
                )
                
                rendered_image = rendered_results['image'] #(B,V,3,H,W)
                rendered_depth = rendered_results['depth'] #(B,V,1,H,W)

                gt_images = batch['output']['imgs'].to(accelerator.device)
                gt_sparse_depth = batch['output']["sparse_gts"].to(accelerator.device).unsqueeze(2)
                    
                current_left_psnr, current_left_ssim = compute_psnr_ssim(pred=rendered_image[:,0,:,:,:],
                                                                         target=gt_images[:,0,:,:,:]
                                                                         )
                current_right_psnr, current_right_ssim = compute_psnr_ssim(pred=rendered_image[:,1,:,:,:],
                                                                         target=gt_images[:,1,:,:,:]
                                                                         )
                
                current_left_mae, current_left_mse = compute_depth_mae_mse(depth_pred=rendered_depth[:,0,:,:,:],
                                                                           depth_gt=gt_sparse_depth[:,0,:,:,:])

                current_right_mae, current_right_mse = compute_depth_mae_mse(depth_pred=rendered_depth[:,1,:,:,:],
                                                                           depth_gt=gt_sparse_depth[:,1,:,:,:])
                
                
                renderd_left_image_metrics.update(psnr=current_left_psnr.data.item(),
                                                  ssim=current_left_ssim.data.item(),
                                                  mae=current_left_mae.data.item(),
                                                  mse=current_left_mse.data.item())
                
                rendered_right_image_metrics.update(psnr=current_right_psnr.data.item(),
                                                  ssim=current_right_ssim.data.item(),
                                                  mae=current_right_mae.data.item(),
                                                  mse=current_right_mse.data.item()
                )
                

                if args.output_vis:
                    
                    # saved rendered images
                    rendered_image_left = rendered_image[0][0].permute(1,2,0) #(H,W,3)
                    rendered_image_right = rendered_image[0][1].permute(1,2,0) #(H,W,3)
                    rendered_image_for_vis = torch.cat((rendered_image_left,rendered_image_right),dim=1).cpu().numpy() 
                    skimage.io.imsave(saved_rendered_image_name,(rendered_image_for_vis*255).astype(np.uint8))

                    # saved rendered depths
                    rendered_depth_left = rendered_depth[0][0][0]
                    rendered_depth_right = rendered_depth[0][1][0]
                    rendered_depth_for_vis = torch.cat((rendered_depth_left,rendered_depth_right),dim=1).cpu().numpy()
                    rendered_depth_for_vis = clean_and_clip(rendered_depth_for_vis)
                    rendered_depth_for_vis = convert_depth_to_disp(factor=328.318735,depth=rendered_depth_for_vis)
                    skimage.io.imsave(saved_rendered_depth_name,rendered_depth_for_vis)
                    
                    
                    rendered_images_all_list.append((rendered_image_for_vis*255).astype(np.uint8))
                    rendered_depths_all_list.append(rendered_depth_for_vis)
                    
                rendered_index = rendered_index + 1

        elif args.ablation_type=="GT":
            print("building the inference dataset........")
            all_key_frame_info_list = []
            for input_frame_name in tqdm(all_frames_idx):
                input_annotation_name = input_frame_name.replace("annotations","annotations_simple")
            
                # current_input_infos
                input_infos_list = get_inputs_info(datapath=val_params['datapath'],
                            reso = val_params['resolution'],
                            first_ref=first_key_frame_lidar_to_world_pose,
                            simple_annotation_path_list=[input_annotation_name],
                            depth_info_params =val_params['depth_info_dict'],
                            extra_list=[])
                
                input_key_frame_info = input_infos_list[0]
                
                all_key_frame_info_list.append(input_key_frame_info)

            configuration_fuse_dataset_for_eval = FusionConfigurationDataset(all_key_frame_info_list)
            configuration_fuse_dataloader_for_eval = DataLoader(configuration_fuse_dataset_for_eval, batch_size=1, shuffle=False)
            configuration_fuse_dataloader_for_eval = accelerator.prepare(
                                            configuration_fuse_dataloader_for_eval)

            # Rendered Images and the Evaluations
            rendered_index = 0
            volumefusion_renderer = my_model.renderer
            for batch in tqdm(configuration_fuse_dataloader_for_eval):
                current_annotation_path = all_frames_idx[rendered_index]
                
                saved_rendered_image_name = current_annotation_path.replace("annotations","rendered_images").replace(".json",".png")
                saved_rendered_image_name = os.path.join(output_saved_folder_path,saved_rendered_image_name)
                os.makedirs(os.path.dirname(saved_rendered_image_name),exist_ok=True)

                saved_rendered_depth_name = current_annotation_path.replace("annotations","rendered_depth").replace(".json",".png")
                saved_rendered_depth_name = os.path.join(output_saved_folder_path,saved_rendered_depth_name)
                os.makedirs(os.path.dirname(saved_rendered_depth_name),exist_ok=True)


                rendered_image = batch['output']['imgs'].to(accelerator.device)
                rendered_depth = batch['output']["sparse_gts"].to(accelerator.device).unsqueeze(2)

                if args.output_vis:
                    
                    # saved rendered images
                    rendered_image_left = rendered_image[0][0].permute(1,2,0) #(H,W,3)
                    rendered_image_right = rendered_image[0][1].permute(1,2,0) #(H,W,3)
                    rendered_image_for_vis = torch.cat((rendered_image_left,rendered_image_right),dim=1).cpu().numpy()
                    
                    skimage.io.imsave(saved_rendered_image_name,(rendered_image_for_vis*255).astype(np.uint8))

                    # saved rendered depths
                    rendered_depth_left = rendered_depth[0][0][0]
                    rendered_depth_right = rendered_depth[0][1][0]
                    rendered_depth_for_vis = torch.cat((rendered_depth_left,rendered_depth_right),dim=1).cpu().numpy()
                    rendered_depth_for_vis = clean_and_clip(rendered_depth_for_vis)
                    rendered_depth_for_vis = convert_depth_to_disp(factor=328.318735,depth=rendered_depth_for_vis)
                    skimage.io.imsave(saved_rendered_depth_name,rendered_depth_for_vis)
                    
                    rendered_images_all_list.append((rendered_image_for_vis*255).astype(np.uint8))
                    rendered_depths_all_list.append(rendered_depth_for_vis)
                    
                rendered_index = rendered_index + 1
            
            
        if args.output_vis:
            images_to_video(image_list=rendered_images_all_list,
                            output_path=saved_rendered_image_video_path,
                            fps=10)
            
            images_to_video(image_list=rendered_depths_all_list,
                            output_path=saved_rendered_depth_video_path,
                            fps=10)
        
          
        saved_quality_results_dict_this_semi_global_map = dict(
            rendered_left=renderd_left_image_metrics.get_stats(),
            rendered_right=rendered_right_image_metrics.get_stats()
        )
        
        saved_into_json(saved_quality_results_dict_this_semi_global_map,path=saved_rendered_results_json_path)

        
        quit()
        
def load_dict_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_dict_to_pickle(dictionary, filepath):
    """
    将字典保存为 pickle 文件

    参数:
        dictionary (dict): 要保存的字典
        filepath (str): 保存的文件路径，例如 'data/my_dict.pkl'
    """
    with open(filepath, 'wb') as f:
        pickle.dump(dictionary, f)

def get_mean(list):
    return sum(list)*1.0/len(list)

import matplotlib.pyplot as plt
import skimage.io
from scipy.spatial.transform import Rotation as Rscipy

def transform_positions(positions, c2w):
    """
    将 G2 的 mean3D 位置变换到 G1 世界坐标系。
    positions: (N, 3)
    c2w: (4, 4)
    """
    N = positions.shape[0]
    homo_positions = torch.cat([positions, torch.ones(N, 1, device=positions.device)], dim=-1)  # (N, 4)
    transformed = (c2w @ homo_positions.T).T[:, :3]  # (N, 3)
    return transformed

def quaternion_multiply(q1, q2):
    """
    四元数乘法：q = q1 * q2
    q1, q2: (N, 4)  [w, x, y, z]
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=1)

def transform_quaternions(q_old, c2w):
    """
    将 G2 的旋转四元数变换到 G1 坐标系。
    q_old: (N, 4) [w, x, y, z]
    c2w: (4, 4)
    """
    R = c2w[:3, :3]  # 提取旋转部分
    R_c2w = Rscipy.from_matrix(R.cpu().numpy())
    q_c2w = R_c2w.as_quat()  # [x, y, z, w]
    q_c2w = torch.tensor([q_c2w[3], q_c2w[0], q_c2w[1], q_c2w[2]], device=q_old.device)  # 转为 [w, x, y, z]
    q_c2w = q_c2w.unsqueeze(0).repeat(q_old.shape[0], 1)  # (N, 4)
    return quaternion_multiply(q_c2w, q_old)

def transform_gs_to_given_pose(g2, c2w):
    """
    输入：
        g1: [1, N1, 14]，主参考坐标系中的高斯
        g2: [1, N2, 14]，待变换到 g1 坐标系的高斯
        c2w: [4, 4]，将 g2 的高斯变换到 g1 坐标系
    输出：
        merged: [1, N1 + N2, 14]，融合后的高斯组
    """
    g2 = g2.squeeze(0)  # -> (N2, 14)

    mean3D = g2[:, 0:3]
    rgb = g2[:, 3:6]
    opacity = g2[:, 6:7]
    quat = g2[:, 7:11]
    scale = g2[:, 11:14]
    
    # 坐标系变换
    mean3D_new = transform_positions(mean3D, c2w)     # (N2, 3)
    quat_new = transform_quaternions(quat, c2w)       # (N2, 4)
    # quat_new = quat     # (N2, 4)
    
    
    g2_transformed = torch.cat([mean3D_new, rgb, opacity, quat_new, scale], dim=1).unsqueeze(0)  # (1, N2, 14)
    return g2_transformed

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

def clean_and_clip(array):
    """
    将 array 中的 NaN 和 Inf 替换为 0，然后 clip 到 [0, 100]
    """
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.clip(array, 0, 100)
    return array

# def remove_gaussians_in_frustum(gaussians, c2w, intrinsics, image_size):
    
#     gaussians = gaussians.float()
#     c2w = c2w.float()
#     intrinsics = intrinsics.float()
    
#     """
#     使用 PyTorch 删除落入两个相机视锥内的高斯点
#     参数:
#         gaussians: (N, 14) or (N, 3) tensor，前3列为高斯中心点
#         c2w: (2, 4, 4) tensor，相机姿态
#         intrinsics: (2, 3, 3) tensor，相机内参
#         image_size: (H, W)，图像大小
#     返回:
#         (M, 14) or (M, 3) tensor，保留不在任一视锥内的高斯点
#     """
#     device = gaussians.device
#     points = gaussians[:, :3]  # (N, 3)
#     N = points.shape[0]
#     H, W = image_size
#     keep_mask = torch.ones(N, dtype=torch.bool, device=device)

#     for i in range(2):
#         w2c = torch.linalg.inv(c2w[i])            # (4, 4)
#         K = intrinsics[i]                         # (3, 3)

#         # 齐次坐标变换
#         homo_points = torch.cat([points, torch.ones((N, 1), device=device)], dim=1)  # (N, 4)
#         cam_points = (w2c @ homo_points.T).T[:, :3]  # (N, 3)

#         # 相机前方
#         in_front = cam_points[:, 2] > 0.2

#         # 像素投影
#         proj = (K @ cam_points.T).T  # (N, 3)
#         proj_x = proj[:, 0] / proj[:, 2]
#         proj_y = proj[:, 1] / proj[:, 2]

#         in_image = (proj_x >= 1) & (proj_x < W) & (proj_y >= 1) & (proj_y < H)
#         visible = in_front & in_image
#         keep_mask &= ~visible  # 可见点设为 False

#     return gaussians[keep_mask]


def add_pitch_to_c2w(c2w, degrees=-5.0):
    pitch_rad = torch.deg2rad(torch.tensor(degrees))
    rot_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(pitch_rad), -torch.sin(pitch_rad)],
        [0, torch.sin(pitch_rad), torch.cos(pitch_rad)],
    ], dtype=c2w.dtype, device=c2w.device)

    # 添加 pitch 旋转（前乘）
    c2w_rotated = c2w.clone()
    c2w_rotated[:3, :3] = rot_x @ c2w[:3, :3]
    return c2w_rotated

def remove_gaussians_in_frustum(gaussians, c2w, intrinsics, image_size):
    """
    角度 + 距离双重判断，移除两个相机视锥内的高斯点
    """
    gaussians = gaussians.float()
    c2w = c2w.float()
    intrinsics = intrinsics.float()

    device = gaussians.device
    points = gaussians[:, :3]
    N = points.shape[0]
    H, W = image_size
    keep_mask = torch.ones(N, dtype=torch.bool, device=device)

    for i in range(2):
        fx = intrinsics[i, 0, 0].float()
        fy = intrinsics[i, 1, 1].float()

        fov_x = 2 * torch.atan(W / (2 * fx))
        fov_y = 2 * torch.atan(H / (2 * fy))

        w2c = torch.linalg.inv(c2w[i])
        homo_points = torch.cat([points, torch.ones((N, 1), device=device)], dim=1)
        cam_points = (w2c @ homo_points.T).T[:, :3]

        x, y, z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
        

        # valid 
        in_front = z > 4.8
        
        # in_front_big = z<30
        
        

        x_angle = torch.atan2(x, z)
        y_angle = torch.atan2(y, z)
        in_fov = (x_angle.abs() <= fov_x / 2) & (y_angle.abs() <= fov_y / 2)

        # visible = in_front & in_fov
        visible = in_front & in_fov
        keep_mask &= ~visible
        # keep_mask = keep_mask & in_front_big
        keep_mask = keep_mask 

    return gaussians[keep_mask]

def images_to_video(image_list, output_path, fps=30):
    """
    Convert a list of images (as numpy arrays) to a video file.

    Args:
        image_list (list of np.ndarray): List of images (H, W, 3), all must have same shape
        output_path (str): Output video file path (e.g. 'output.mp4')
        fps (int): Frames per second
    """
    if len(image_list) == 0:
        raise ValueError("Image list is empty.")
    
    height, width, _ = image_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in image_list:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR

    writer.release()
    print(f"✅ Video saved to: {output_path}")

def compute_psnr_ssim(pred, target):

    B, C, H, W = pred.shape

    pred = pred
    target = target
    psnr_val = psnr(pred, target, data_range=1.0)
    ssim_val = ssim(pred, target, data_range=1.0)

    return psnr_val,ssim_val
    # return torch.stack(psnr_vals).mean().data.item(), torch.stack(ssim_vals).mean().data.item()

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



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path')
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--semi_global_map', type=str)
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--ablation_type', type=str, default='')

    parser.add_argument(
        "--output_vis",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    ) # visualize the outputs image
    
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    main(args)