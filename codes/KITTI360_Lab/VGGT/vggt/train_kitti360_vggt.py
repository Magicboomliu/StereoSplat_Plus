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







