import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import os
from tqdm import tqdm

def save_dict_to_json(data_dict, file_path, indent=4):
    """
    Save a Python dictionary to a JSON file.

    Args:
        data_dict (dict): The dictionary to save.
        file_path (str): Output file path (e.g., "output.json").
        indent (int): Indentation level for readability (default: 4).
    """
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=indent)

if __name__=="__main__":
    
    
    
    parser = argparse.ArgumentParser(description='Metric3D-V2 Dataset Configration')    
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--out_path', type=str, default='./vis_depth')
    parser.add_argument('--sequence_name', type=str, default='2013_05_28_drive_0000_sync')
    args = parser.parse_args()

    os.makedirs(args.out_path,exist_ok=True)
    
    test_annotations_dict = dict()
    test_annotations_dict['files'] = []
        
    if args.sequence_name=="All":
        saved_test_annotations_json_path = os.path.join(args.out_path,"test_annotations.json")
        sequence_list = os.listdir(os.path.join(args.root_path,"data_2d_raw/"))
        
        for seq in sorted(sequence_list):
            input_image_folder_left = os.path.join(args.root_path,"data_2d_raw/",seq,"image_00/data_rect")
            input_image_folder_right = os.path.join(args.root_path,"data_2d_raw/",seq,"image_01/data_rect")
            
            print("currently processing sequence name is :", seq)

            for image_fname in tqdm(os.listdir(input_image_folder_left)):
                input_image_fname_left = os.path.join(input_image_folder_left,image_fname)
                input_image_fname_right = input_image_fname_left.replace("image_00","image_01")
                assert os.path.exists(input_image_fname_left)
                assert os.path.exists(input_image_fname_right)
                
                input_image_depth_left = input_image_fname_left.replace("data_2d_raw","projected_sparse_lidar/data_2d_raw")
                assert os.path.exists(input_image_depth_left)
                input_image_depth_right = input_image_fname_right.replace("data_2d_raw","projected_sparse_lidar/data_2d_raw")
                assert os.path.exists(input_image_depth_right)
                
    
                left_image_dict = dict()
                right_image_dict = dict()
                
                left_image_dict['rgb'] = input_image_fname_left
                left_image_dict['depth'] = input_image_depth_left
                left_image_dict['depth_scale'] = 256
                left_image_dict['cam_in'] = [552.554261,552.554261,682.049453,238.769549]
                
                right_image_dict['rgb'] = input_image_fname_right
                right_image_dict['depth'] = input_image_depth_right
                right_image_dict['depth_scale'] = 256
                right_image_dict['cam_in'] = [552.554261,552.554261,682.049453,238.769549]
                
                test_annotations_dict['files'].append(left_image_dict)
                test_annotations_dict['files'].append(right_image_dict)
    
    else:
        saved_test_annotations_json_path = os.path.join(args.out_path,"test_annotations_{}.json".format(args.sequence_name))
        input_image_folder_left = os.path.join(args.root_path,"data_2d_raw/",args.sequence_name,"image_00/data_rect")
        input_image_folder_right = os.path.join(args.root_path,"data_2d_raw/",args.sequence_name,"image_01/data_rect")

        for image_fname in tqdm(os.listdir(input_image_folder_left)):
            input_image_fname_left = os.path.join(input_image_folder_left,image_fname)
            input_image_fname_right = input_image_fname_left.replace("image_00","image_01")
            assert os.path.exists(input_image_fname_left)
            assert os.path.exists(input_image_fname_right)
            
            input_image_depth_left = input_image_fname_left.replace("data_2d_raw","projected_sparse_lidar/data_2d_raw")
            assert os.path.exists(input_image_depth_left)
            input_image_depth_right = input_image_fname_right.replace("data_2d_raw","projected_sparse_lidar/data_2d_raw")
            assert os.path.exists(input_image_depth_right)
            
            

            left_image_dict = dict()
            right_image_dict = dict()
            
            left_image_dict['rgb'] = input_image_fname_left
            left_image_dict['depth'] = input_image_depth_left
            left_image_dict['depth_scale'] = 256
            left_image_dict['cam_in'] = [552.554261,552.554261,682.049453,238.769549]
            
            right_image_dict['rgb'] = input_image_fname_right
            right_image_dict['depth'] = input_image_depth_right
            right_image_dict['depth_scale'] = 256
            right_image_dict['cam_in'] = [552.554261,552.554261,682.049453,238.769549]
            
            test_annotations_dict['files'].append(left_image_dict)
            test_annotations_dict['files'].append(right_image_dict)
    
    save_dict_to_json(test_annotations_dict,saved_test_annotations_json_path)


            
            
    
    
    