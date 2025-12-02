import json
import os
import numpy as np
import random
from tqdm import tqdm


def save_into_json(data_dict, filename):
    with open(filename, 'w') as f:
        json.dump(data_dict, f)



if __name__=="__main__":
    
    root_path = "/data4/zliu/Difix3D/KITTI360_Degradtations_Pairs/"
    
    saved_trainval_json_path = "/home/zliu/Project2025/OneStageTraining/FeedStereoGS/filenames/kitti360/difix_dataset/trainval.json"
    os.makedirs(os.path.dirname(saved_trainval_json_path),exist_ok=True)
        
    input_images_root_path = os.path.join(root_path, "image")
    target_images_root_path = os.path.join(root_path, "target_image")
    ref_images_root_path = os.path.join(root_path, "ref_image")
    
    assert os.path.exists(input_images_root_path)
    assert os.path.exists(target_images_root_path)
    assert os.path.exists(ref_images_root_path)
    
    degradation_level_list = ["0","1","2"]
    
    train_val_filename_dict = {}
    train_val_filename_dict['train'] = dict()
    train_val_filename_dict['test'] = dict()
    
    
    
    current_idx = 0
    
    for level in degradation_level_list:
        
        for fname in tqdm(sorted(os.listdir(os.path.join(input_images_root_path, level)))):
            
            input_image_path = os.path.join(input_images_root_path, level, fname)
            target_image_path = os.path.join(target_images_root_path, level, fname)
            ref_image_path = os.path.join(ref_images_root_path, level, fname)
            
            assert os.path.exists(input_image_path)
            assert os.path.exists(target_image_path)
            assert os.path.exists(ref_image_path)
            
            current_data_dict = {}
            current_data_dict['image'] = input_image_path
            current_data_dict['target_image'] = target_image_path
            current_data_dict['ref_image'] = ref_image_path
            current_data_dict['prompt'] = "remove degradation"


            
            if current_idx%10==0:
                train_val_filename_dict['test'][current_idx] = current_data_dict
            else:
                train_val_filename_dict['train'][current_idx] = current_data_dict
            
            current_idx += 1
        
    
    save_into_json(train_val_filename_dict, saved_trainval_json_path)
    
    
    
    
    





