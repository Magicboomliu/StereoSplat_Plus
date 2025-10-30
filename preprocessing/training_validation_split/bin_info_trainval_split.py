import os
import numpy as np
import sys
import re
import random
import json
import pickle as pkl
from tqdm import tqdm


def extract_2013_to_sync(s: str) -> str | None:
    """
    从字符串中提取以 '2013' 开始、以 'sync' 结束的最短片段（含两端）。
    找不到则返回 None。
    """
    m = re.search(r'2013.*?sync', s)
    return m.group(0) if m else None

def save_into_txt(list, path):  
    with open(path,'w') as f:
        for idx, item in enumerate(list):
            if idx!=len(list)-1:
                f.writelines(item+"\n")
            else:
                f.writelines(item)

def load_pkl_file(path):
    with open(path, 'rb') as f:
        data_dict = pkl.load(f)
    return data_dict



# def remove_the_bins_less_than_six_views(bin_path):
#     for fname in sorted(os.listdir(bin_path)):
#         bin_info = json.load(open(os.path.join(bin_path,fname)))
#         if len(bin_info['sensor_info']['LIDAR_TOP'])<6:
#             os.remove(os.path.join(bin_path,fname))



if __name__=="__main__":
    bin_path = "/data1/StereoDatasets/KITTI/KITTI360/feedforward_bins/bin_infos_8.0_FirstLIDAR"
    # remove all the dynamic sequnces
    dynamic_sequences = ["2013_05_28_drive_0003_sync", "2013_05_28_drive_0007_sync","2013_05_28_drive_0010_sync"]
    
    all_valid_sequences = []
    train_valiation_complete_path = "/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/train_complete"
    
    for fname in tqdm(sorted(os.listdir(bin_path))):
        sequence_name = extract_2013_to_sync(fname)
        if sequence_name in dynamic_sequences:
            continue
        
        bin_pickle_files_path = os.path.join(bin_path,fname)
        assert os.path.exists(bin_pickle_files_path)
        
        bin_info = load_pkl_file(bin_pickle_files_path)
        if len(bin_info['sensor_info']['CAM_LEFT'])>=6:
            all_valid_sequences.append(fname)
    
    # 随机划分 90/10
    random.seed(42)  # 固定随机种子以复现
    random.shuffle(all_valid_sequences)
    n_total = len(all_valid_sequences)
    n_val = max(1, int(n_total * 0.10)) if n_total > 0 else 0
    n_train = n_total - n_val

    train_set = all_valid_sequences[:n_train]
    val_set   = all_valid_sequences[n_train:]
    
    
    save_into_txt(train_set, os.path.join(train_valiation_complete_path, 'train.txt'))
    save_into_txt(val_set, os.path.join(train_valiation_complete_path, 'val.txt'))
    
    save_into_txt(all_valid_sequences, os.path.join(train_valiation_complete_path, 'all.txt'))
