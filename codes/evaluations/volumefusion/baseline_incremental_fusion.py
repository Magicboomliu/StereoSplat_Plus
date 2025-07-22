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

from model.utils.image import resize_image,HWC3

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
    
    sys.path.append("..")
    
    
    
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


    
    # Define the Model/Optimizer/Schduler Here
    my_model = VolumeFusion(backbone=cfg.model.backbone,
                            neck=cfg.model.neck,
                            costvolume_gs=cfg.model.costvolume_gs,
                            volume_gs=cfg.model.volume_gs,
                            losses_params=cfg.model.losses_params,
                            camera_args=cfg.camera_args,
                            dataset_params=cfg.dataset_params,
                            use_checkpoint=cfg.use_checkpoint)

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
    
    my_model= accelerator.prepare(
        my_model
    )
    
    
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

        
        print("Step 2: begin incremenatl gaussain fusion.....")
        
        with torch.no_grad():
            my_model.eval()
            for batch in tqdm(configuration_fuse_dataloader):
                print(batch['input']['imgs'].shape)
                quit()
                

        


def get_mean(list):
    return sum(list)*1.0/len(list)
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path')
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--semi_global_map', type=str)
    parser.add_argument('--pretrained_model_path', type=str, default='')

    parser.add_argument(
        "--output_vis",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    ) # visualize the outputs image
    
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    main(args)