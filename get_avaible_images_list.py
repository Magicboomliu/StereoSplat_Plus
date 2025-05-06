import os
import json
import random
import operator
import functools
import itertools
import multiprocessing

import torch
import torchvision
import numpy as np
import skimage
import pycocotools.mask


if __name__=="__main__":
    
    
    kitti360_root_path = "/media/zliu/data12/dataset/KITTI/KITTI360"
    example_seq = "2013_05_28_drive_0000_sync"
    
    avaiable_list = []
    
    
    # "/media/zliu/data12/dataset/KITTI/KITTI360/annotations/2013_05_28_drive_0000_sync/image_00/data_rect/"
    annotations_root_folder = os.path.join(kitti360_root_path,"annotations",example_seq)
    annotations_folder = os.path.join(annotations_root_folder,"image_00/data_rect/")
    
    for idx, fname in enumerate(sorted(os.listdir(annotations_folder))):
        
        current_annotations_path = os.path.join("annotations/{}/image_00/data_rect".format(example_seq),fname)
        current_annotations_path_right = current_annotations_path.replace("image_00","image_01")
        
        assert os.path.exists(os.path.join(kitti360_root_path,current_annotations_path))
        assert os.path.exists(os.path.join(kitti360_root_path,current_annotations_path_right))
        
        avaiable_list.append(current_annotations_path)

    
    
    with open("{}_list.txt".format(example_seq),'w') as f:
        for idx, fname in enumerate(avaiable_list):
            if idx!=len(avaiable_list)-1:
                f.writelines(fname+"\n")
            else:
                f.writelines(fname)