import numpy as np
import json
import os


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    
    input_json_path = "/home/zliu/Project2025/OneStageTraining/FeedStereoGS/filenames/kitti360/difix_dataset/trainval.json"
    
    loaded_dict_data = read_json(input_json_path)['train']
    
    print(len(loaded_dict_data.keys()))

    
    