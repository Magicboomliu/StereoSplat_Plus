import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import skimage.io

def normalized_disp(disp):
    range = np.minimum(disp.max() / (disp.min() + 0.001), 50.0)
    max = disp.max()
    min = max / range
    depth = 1 / np.maximum(disp, min)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) # range from 0 ~1
    
    return depth


def save_results_into_uint16(results,scale_factor=50):
    
    results = results * scale_factor
    results = results.astype(np.uint16)
    
    return results
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')    
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--out_path', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--sequence_name', type=str, default='2013_05_28_drive_0000_sync')
    parser.add_argument('--checkpoint_path', type=str, default="/data1/zliu/pretrained_foundataion_models/depth_estimation/DepthAnythingV2/depth_anything_v2_vitl.pth")
    
    args = parser.parse_args()

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    encoder = args.encoder
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))
    model = model.to(DEVICE).eval()
    print("Loaded the Models")
    
    os.makedirs(args.out_path,exist_ok=True)
    
    
    if args.sequence_name=="All":
        sequence_list = os.listdir(args.root_path,"data_2d_raw/")
        for seq in sorted(sequence_list):
            input_image_folder_left = os.path.join(args.root_path,"data_2d_raw/",seq,"image_00/data_rect")
            input_image_folder_right = os.path.join(args.root_path,"data_2d_raw/",seq,"image_01/data_rect")
            
            for image_fname in tqdm(os.listdir(input_image_folder_left)):
                input_image_fname_left = os.path.join(input_image_folder_left,image_fname)
                input_image_fname_right = input_image_fname_left.replace("image_00","image_01")
                assert os.path.exists(input_image_fname_left)
                assert os.path.exists(input_image_fname_right)
                
                raw_img_left = cv2.imread(input_image_fname_left)
                raw_img_right = cv2.imread(input_image_fname_right)
                
                with torch.no_grad():
                    
                    est_depth_left = model.infer_image(raw_img_left)
                    est_depth_right = model.infer_image(raw_img_right)
                    
                    
                    est_depth_left_uint16 = save_results_into_uint16(est_depth_left)
                    est_depth_right_uint16 = save_results_into_uint16(est_depth_right)
                    
                    
                    saved_left_image_path = input_image_fname_left.replace(args.root_path,args.out_path)
                    saved_right_image_path = input_image_fname_right.replace(args.root_path,args.out_path)
                    
                    os.makedirs(os.path.dirname(saved_left_image_path),exist_ok=True)
                    os.makedirs(os.path.dirname(saved_right_image_path),exist_ok=True)
                    
                    skimage.io.imsave(saved_left_image_path,est_depth_left_uint16)
                    skimage.io.imsave(saved_right_image_path,est_depth_right_uint16)
        
    else:
        input_image_folder_left = os.path.join(args.root_path,"data_2d_raw/",args.sequence_name,"image_00/data_rect")
        input_image_folder_right = os.path.join(args.root_path,"data_2d_raw/",args.sequence_name,"image_01/data_rect")
        
        for image_fname in tqdm(os.listdir(input_image_folder_left)):
            input_image_fname_left = os.path.join(input_image_folder_left,image_fname)
            input_image_fname_right = input_image_fname_left.replace("image_00","image_01")
            assert os.path.exists(input_image_fname_left)
            assert os.path.exists(input_image_fname_right)
            
            raw_img_left = cv2.imread(input_image_fname_left)
            raw_img_right = cv2.imread(input_image_fname_right)
            
            with torch.no_grad():
                
                est_depth_left = model.infer_image(raw_img_left)
                est_depth_right = model.infer_image(raw_img_right)
                
                
                est_depth_left_uint16 = save_results_into_uint16(est_depth_left)
                est_depth_right_uint16 = save_results_into_uint16(est_depth_right)
                
                
                saved_left_image_path = input_image_fname_left.replace(args.root_path,args.out_path)
                saved_right_image_path = input_image_fname_right.replace(args.root_path,args.out_path)
                
                os.makedirs(os.path.dirname(saved_left_image_path),exist_ok=True)
                os.makedirs(os.path.dirname(saved_right_image_path),exist_ok=True)
                
                skimage.io.imsave(saved_left_image_path,est_depth_left_uint16)
                skimage.io.imsave(saved_right_image_path,est_depth_right_uint16)
                
    
    print("All Finsihed!")
        
    
    
    # for cam_idx, cam_types in enumerate(os.listdir(args.inputdir)):
    #     if "CAM" in cam_types:
    #         output_cam_folder = os.path.join(args.outdir,cam_types)
    #         os.makedirs(output_cam_folder,exist_ok=True)
    #         print("Current Processed {}: {}/{}".format(cam_types,cam_idx,6))
    #         for fname in tqdm(os.listdir(os.path.join(args.inputdir,cam_types))):
    #             input_image_fname = os.path.join(args.inputdir,cam_types,fname)
    #             assert os.path.exists(input_image_fname)

    #             raw_img = cv2.imread(input_image_fname)
                
    #             with torch.no_grad():
    #                 est_depth = model.infer_image(raw_img)
                
    #             saved_name = os.path.join(output_cam_folder,os.path.basename(input_image_fname))
    #             saved_name = saved_name.replace(".jpg",".npy")
    #             np.save(saved_name,est_depth)
            
    
    # print("All Processed Done for the input {}".format(args.inputdir))

    