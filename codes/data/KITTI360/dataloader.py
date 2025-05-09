import os
import os.path as osp
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import json
import random
import pickle as pkl
from functools import cached_property
from pathlib import Path
import imageio.v2 as imageio
import glob
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import cv2
import copy
from io import BytesIO
from einops import rearrange, repeat, einsum
cv2.setNumThreads(0) 
cv2.ocl.setUseOpenCL(False)

import sys
sys.path.append("../..")
from model.utils.image import resize_image, HWC3
from model.utils.typing import *
from model.utils.camera import get_camera, rescale_intrisic
from model.utils.ops import get_cam_info_gaussian, get_ray_directions, get_rays


class KITTI360(Dataset):
    def __init__(
        self,
        resolution: list = [224, 400],
        split: str = "train",
        use_center: bool = True,
        use_first: bool = False,
        use_last: bool = False):
        super().__init__()
        


    def __getitem__(self, index):

        pass

    def __len__(self):
        return 



if __name__=="__main__":
    pass