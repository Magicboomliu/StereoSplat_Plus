import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import json

from tqdm import tqdm
import argparse

from model_no_ref import Difix_No_Ref
from model_ref import DifixRef

from PIL import Image
import skimage.io





def read_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    

    # Argument parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_test_filename', type=str, default=None, help='Path to the dataset')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a model state dict to be used')

    parser.add_argument('--height', type=int, default=576, help='Height of the input image')
    parser.add_argument('--width', type=int, default=1024, help='Width of the input image')
    
    
    parser.add_argument('--ablation_study_name', type=str, default=None, help='The prompt to be used')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    parser.add_argument("--use_model_type", type=str, default="huggingface or local", help='')
    parser.add_argument("--output_folder", type=str, default=None, help='Directory to save the output')
    parser.add_argument("--vis", action='store_true', help='If the input is a video')
    parser.add_argument("--use_ref", action='store_true', help='If the input is a video')
    args = parser.parse_args()
    
    
    # loading the model here
    if args.use_model_type == "huggingface":
        model_name = args.model_name
        model_path = None
    elif args.use_model_type == "local":
        if args.use_ref:
            model_name = "nvidia/difix_ref"
        else:
            model_name = "nvidia/difix"
        model_path = args.model_path
    

    if args.use_ref:
        
        model = DifixRef(
            pretrained_name=model_name,
            pretrained_path=model_path,
            timestep=args.timestep,
            mv_unet=True if args.use_ref else False,
        )
        # set the eval mode.
        model.set_eval()
    else:
        # Initialize the model
        model = Difix_No_Ref(
            pretrained_name=args.model_name,
            pretrained_path=args.model_path,
            timestep=args.timestep,
            mv_unet=True if args.use_ref  else False,
        )
        # set the eval mode.
        model.set_eval()
    
    print("loading the model successfully......")
    
    

    dataset_path = args.input_test_filename    
    dataset_files = read_json_file(dataset_path)['test']
    
    
    for data_id, data_item in tqdm(dataset_files.items()):
    
        current_input_image_id = data_id 
        current_input_image_path = data_item['input_image']
        current_output_image_path = data_item['output_image']
        current_ref_image_path = data_item['ref_image']
        current_prompt = data_item['prompt']
        
        
        current_input_image = Image.open(current_input_image_path).convert('RGB')
        current_ref_image = Image.open(current_ref_image_path).convert('RGB') if args.use_ref else None
        
        
        print(current_input_image.size)
        print(current_ref_image.size)
        

        quit()