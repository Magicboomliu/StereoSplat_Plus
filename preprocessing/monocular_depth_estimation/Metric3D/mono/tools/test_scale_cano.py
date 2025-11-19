import os
import os.path as osp
import cv2
import time
import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)
import argparse
import mmcv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from datetime import timedelta
import random
import numpy as np
from mono.utils.logger import setup_logger
import glob
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_scalecano_test_with_custom_data
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_from_annos, load_data

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--show-dir', help='the dir to save logs and visualization results')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nnodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--launcher', choices=['None', 'pytorch', 'slurm', 'mpi', 'ror'], default='slurm', help='job launcher')
    parser.add_argument('--test_data_path', default='None', type=str, help='the path of test data')
    parser.add_argument('--batch_size', default=1, type=int, help='the batch size for inference')
    args = parser.parse_args()
    return args


from nuscenes.nuscenes import NuScenes
import os
from PIL import Image
import numpy as np

def loaded_nuscene_intrinsic_dict(version="v1.0-mini",root_path="/data2/zliu_backup_data/nuScenes/v1.0-mini/"):
    
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    camera_name_dict = dict()
    for cam_name in nusc.sample[0]['data'].keys():
        if "CAM" in cam_name:
            
            sample_data = nusc.get('sample_data', nusc.sample[0]['data'][cam_name])
            calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            intrinsic = np.array(calib['camera_intrinsic'])
  
            if cam_name not in camera_name_dict.keys():
                camera_name_dict[cam_name] = intrinsic
                '''
                [[809.22099057(fx)   0.         829.21960033(cx)]
                    [  0.         809.22099057(fy) 481.77842385(cy)]
                    [  0.           0.           1.        ]]
                '''
    return camera_name_dict
                


def load_nuscene_data(path,instrinsic_dict):
    
    data_list =[]
    
    for cam_name in os.listdir(path):
        if "CAM" in cam_name:
            current_cam_instrinics = instrinsic_dict[cam_name]
            intrinsic_list = [current_cam_instrinics[0][0],current_cam_instrinics[1][1],current_cam_instrinics[0][2],current_cam_instrinics[1][2]]
            current_sub_folder_path = os.path.join(path,cam_name)
            for fname in os.listdir(current_sub_folder_path):
                current_rgb_path = os.path.join(current_sub_folder_path,fname)
                
                input_dict = dict()
                input_dict['rgb'] = current_rgb_path
                input_dict['depth'] = None
                input_dict['intrinsic'] = intrinsic_list
                input_dict['filename'] = os.path.basename(current_rgb_path)
                input_dict['folder'] = os.path.dirname(current_rgb_path)
                data_list.append(input_dict)
    
    return data_list
            



def main(args):
    
    # loaded the nuscence data, including the camera instrincs
    nuscene_intrinsic_dict = loaded_nuscene_intrinsic_dict()
    
    os.chdir(CODE_SPACE)
    cfg = Config.fromfile(args.config)
    
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        
    # show_dir is determined in this priority: CLI > segment in file > filename
    if args.show_dir is not None:
        # update configs according to CLI args if args.show_dir is not None
        cfg.show_dir = args.show_dir
    else:
        # use condig filename + timestamp as default show_dir if args.show_dir is None
        cfg.show_dir = osp.join('./show_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                args.timestamp)
    
    # ckpt path
    if args.load_from is None:
        raise RuntimeError('Please set model path!')
    cfg.load_from = args.load_from
    cfg.batch_size = args.batch_size
    
    # load data info
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.mldb_info = data_info
    # update check point info
    reset_ckpt_path(cfg.model, data_info)
    
    # create show dir
    os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)
    
    # init the logger before other steps
    cfg.log_file = osp.join(cfg.show_dir, f'{args.timestamp}.log')
    logger = setup_logger(cfg.log_file)
    
    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')
    
    # init distributed env dirst, since logger depends on the dist info
    if args.launcher == 'None':
        cfg.distributed = False
    else:
        cfg.distributed = True
        init_env(args.launcher, cfg)
    logger.info(f'Distributed training: {cfg.distributed}')
    
    
    # quut: log the data
    # dump config 
    cfg.dump(osp.join(cfg.show_dir, osp.basename(args.config)))
    test_data_path = args.test_data_path

    if not os.path.isabs(test_data_path):
        test_data_path = osp.join(CODE_SPACE, test_data_path)

    
    # loaded the nuscence data
    test_data = load_nuscene_data(path=test_data_path,
                      instrinsic_dict=nuscene_intrinsic_dict)
    
    
    if not cfg.distributed:
        main_worker(0, cfg, args.launcher, test_data)
    else:
        # distributed training
        if args.launcher == 'ror':
            local_rank = cfg.dist_params.local_rank
            main_worker(local_rank, cfg, args.launcher, test_data)
        else:
            mp.spawn(main_worker, nprocs=cfg.dist_params.num_gpus_per_node, args=(cfg, args.launcher, test_data))
        
def main_worker(local_rank: int, cfg: dict, launcher: str, test_data: list):
    if cfg.distributed:
        cfg.dist_params.global_rank = cfg.dist_params.node_rank * cfg.dist_params.num_gpus_per_node + local_rank
        cfg.dist_params.local_rank = local_rank

        if launcher == 'ror':
            init_torch_process_group(use_hvd=False)
        else:
            torch.cuda.set_device(local_rank)
            default_timeout = timedelta(minutes=30)
            dist.init_process_group(
                backend=cfg.dist_params.backend,
                init_method=cfg.dist_params.dist_url,
                world_size=cfg.dist_params.world_size,
                rank=cfg.dist_params.global_rank,
                timeout=default_timeout)
    
    logger = setup_logger(cfg.log_file)
    # build model
    model = get_configured_monodepth_model(cfg, )
    
    # config distributed training
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()
        
    # load ckpt
    model, _,  _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    model.eval()

    
    do_scalecano_test_with_custom_data(
        model, 
        cfg,
        test_data,
        logger,
        cfg.distributed,
        local_rank,
        cfg.batch_size,
    )
    
if __name__ == '__main__':
    args = parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.timestamp = timestamp
    main(args)    