import os
import numpy as np
import sys
import re
import random

def split_list(input_list, seed=42,ratio=0.1):
    """
    Split input_list into two lists: 90% (big list) and 10% (small list).
    """
    total_len = len(input_list)
    small_len = max(1, int(ratio * total_len))  # At least 1 element
    random.seed(seed)
    indices = list(range(total_len))
    random.shuffle(indices)

    small_indices = set(indices[:small_len])
    small_list = [input_list[i] for i in small_indices]
    big_list = [input_list[i] for i in range(total_len) if i not in small_indices]

    return big_list, small_list


def split_bins_into_train_and_val(args):
    
    bins_all = sorted(os.listdir(args.feedforward_bin_path))
    
    all_list = []
    train_list = []
    val_list = []
    for bins in bins_all:
        sequence_name = re.search(r"(2013_\d{2}_\d{2}_drive_\d{4}_sync)", bins).group(1)
        if args.sequence_name!='all':
            if sequence_name==args.sequence_name:
                all_list.append(bins)
        else:
            all_list.append(bins)
    
    
    big_list, small_list = split_list(all_list,args.seed,args.val_ratio)
    
    return big_list, small_list, all_list


def save_into_txt(filename_list,path):
    with open(path,'w') as f:
        for idx, fname in enumerate(filename_list):
            if idx!=len(filename_list)-1:
                f.writelines(fname+"\n")
            else:
                f.writelines(fname)

if __name__=="__main__":
    
    import argparse    
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument(
        "--feedforward_bin_path",
        type=str,
        required=False,
        default="/media/zliu/data12/dataset/KITTI/VSRD_Format/",
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "--sequence_name",
        type=str,
        required=False,
        default="/media/zliu/data12/dataset/KITTI/VSRD_Format/",
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=False,
        default="/home/zliu/Project2025/FeedStereoGS/filenames/trainval",
        help="specify the root path of dataset",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=1024,
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="specify the root path of dataset",
    )
    
    args = parser.parse_args()
    os.makedirs(args.output_folder,exist_ok=True)
    
    train_list, val_list, all_list = split_bins_into_train_and_val(args)
    save_into_txt(train_list,os.path.join(args.output_folder,'train_{}.txt'.format(args.sequence_name)))
    save_into_txt(val_list,os.path.join(args.output_folder,'val_{}.txt'.format(args.sequence_name)))
    save_into_txt(all_list,os.path.join(args.output_folder,'all_{}.txt'.format(args.sequence_name)))   