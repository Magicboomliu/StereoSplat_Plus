import os
import numpy as np
import cv2
import skimage.io
from tqdm import tqdm
import random
import json


def save_dict_into_json(data_dict, json_path):
    with open(json_path, "w") as f:
        json.dump(data_dict, f, indent=4)
    

if __name__ == "__main__":
    
    prompt = "remove degradation"
    
    trainval_filelist = "/home/zliu/Project2025/OneStageTraining/FeedStereoGS/filenames/kitti360/difix_dataset/trainval.json"
    
    
    saved_json_dict = {}
    saved_json_dict["train"] = {}
    saved_json_dict["test"] = {}
    
    
    os.makedirs(os.path.dirname(trainval_filelist), exist_ok=True)
    
    
    root_folder = "/data1/zliu/KITTI360_Completed/KITTI360_Degradtations_Pairs/"
    
    input_images_folder = os.path.join(root_folder, "image")
    target_images_folder = os.path.join(root_folder, "target_image")
    ref_images_folder = os.path.join(root_folder, "ref_image")
    
    assert os.path.exists(input_images_folder) 
    assert os.path.exists(target_images_folder) 
    assert os.path.exists(ref_images_folder) 
    
    
    train_val_filelist_all = []
    different_levels_list = ["0", "1", "2"]
    instance_idx = 0
    
    for different_level in different_levels_list:
        print(f"Processing different level: {different_level}")
        
        current_input_images_folder = os.path.join(input_images_folder, different_level)
        current_target_images_folder = os.path.join(target_images_folder, different_level)
        current_ref_images_folder = os.path.join(ref_images_folder, different_level)
        
    
        assert os.path.exists(current_input_images_folder)
        assert os.path.exists(current_target_images_folder)
        assert os.path.exists(current_ref_images_folder)
        
        
        for fname in tqdm(os.listdir(current_input_images_folder)):
            
            current_data_dict = {}

            input_image_path_abs = os.path.join(current_input_images_folder, fname)
            target_image_path_abs = os.path.join(current_target_images_folder, fname)
            ref_image_path_abs = os.path.join(current_ref_images_folder, fname)
            
            current_data_dict["image"] = input_image_path_abs
            current_data_dict["target_image"] = target_image_path_abs
            current_data_dict["ref_image"] = ref_image_path_abs
            current_data_dict["prompt"] = prompt
            
            
            if instance_idx%10==0:
                saved_json_dict["test"][instance_idx] = current_data_dict
            
            else:
                saved_json_dict["train"][instance_idx] = current_data_dict

        
            instance_idx +=1
            
    
    save_dict_into_json(saved_json_dict, trainval_filelist)



        

        
        
        
        
    
    
    

