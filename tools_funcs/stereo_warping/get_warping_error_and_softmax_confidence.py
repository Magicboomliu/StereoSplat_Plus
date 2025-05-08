import os
from warping_numpy import disp_warp_np
from warping_torch import disp_warp
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_the_Metric3DV2_Results(path,scale=256):
    # Read the image in unchanged mode (preserves uint16 format)
    img = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
    img = img/scale
    return img

def loaded_image(path):
    results = np.array(Image.open(path).convert("RGB")).astype(np.float32)
    return results
    

def convert_depth_into_disp(depth,focal_length,baseline,eps=1e-6,largest=320):
    
    disp = focal_length * baseline/(depth+1e-6)
    disp = np.clip(disp,a_min=0, a_max=largest)
    return disp

def get_conf_and_warped_error(left_image_data,right_image_data,metric_depth,focal_length=552.554261,baseline=0.5941836995443964,
                              temp=200):
  
    if np.max(left_image_data) and np.max(right_image_data)>2:
        temp = temp
    else:
        temp = 1
    
    disp = convert_depth_into_disp(metric_depth,focal_length,baseline,eps=1e-6,largest=320)
    

    left_image_data_t = torch.from_numpy(left_image_data).permute(2,0,1).unsqueeze(0)
    
    right_image_data_t = torch.from_numpy(right_image_data).permute(2,0,1).unsqueeze(0)
    
    disp = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0)

    warped_left, valid_mask_left = disp_warp(img=right_image_data_t, disp=disp, padding_mode='border')
    
    error = torch.abs(warped_left-left_image_data_t)
    error = torch.sum(error,dim=1,keepdim=True)
    
    conf =torch.exp(-error/temp)
    
    error = error.squeeze(0).squeeze(0).cpu().numpy()
    conf = conf.squeeze(0).squeeze(0).cpu().numpy()

    return error, conf


if __name__=="__main__":
    
    left_image = "/data1/StereoDatasets/KITTI/KITTI360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000259.png"
    right_image = left_image.replace("image_00","image_01")
    metric_depth = left_image.replace("/data1/StereoDatasets/KITTI/KITTI360/","/data1/StereoDatasets/KITTI/KITTI360/monocular_depth/Metric3DV2/").replace(os.path.basename(left_image)[:-4],"{}_dpt".format(os.path.basename(left_image)[:-4]))
    
    assert os.path.exists(left_image)
    assert os.path.exists(right_image)
    assert os.path.exists(metric_depth)
    
    
    left_image_data = loaded_image(left_image)
    right_image_data = loaded_image(right_image)
    metric_depth = load_the_Metric3DV2_Results(metric_depth)
  

    error,conf= get_conf_and_warped_error(left_image_data,right_image_data,metric_depth,focal_length=552.554261,baseline=0.5941836995443964,
                              temp=100)
    
    print(conf.shape)
    print(conf.max())
    print(conf.min())
    print(conf.mean())
        
    # import skimage.io 
    # warped_left_vis = warped_left.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
    # skimage.io.imsave("warped.png",warped_left_vis)   
    # skimage.io.imsave("left.png",left_image_data.astype(np.uint8)) 
    # skimage.io.imsave("right.png",right_image_data.astype(np.uint8)) 

    
        
    # pass