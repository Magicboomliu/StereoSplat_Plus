import os
import numpy as np
import sys
from tqdm import tqdm
import pickle
from pathlib import Path
from typing import Any, Union
import pycocotools.mask
import torch
import json

def read_annotation(annotation_filename):

    with open(annotation_filename) as file:
        annotation = json.load(file)
    intrinsic_matrix = torch.as_tensor(annotation["intrinsic_matrix"])
    extrinsic_matrix = torch.as_tensor(annotation["extrinsic_matrix"])
    
    return intrinsic_matrix,extrinsic_matrix



def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def load_pickle_file(filepath: Union[str, Path]) -> Any:
    """
    读取 pickle 文件并返回其内容。
    
    Args:
        filepath (str or Path): .pkl 或 .pickle 文件路径
    
    Returns:
        Any: 文件中保存的 Python 对象
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with filepath.open("rb") as f:
        data = pickle.load(f)
    
    return data


def save_dict_to_json(data_dict, filename):
    """
    Save a dictionary to a JSON file.

    Args:
        data_dict (dict): The dictionary to save.
        filename (str): The path to the JSON file to create.
    """
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)


if __name__=="__main__":
    
    root_path = "/data1/StereoDatasets/KITTI/KITTI360/"
    annotation_list_path = "/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/avaliable_lists/2013_05_28_drive_0000_sync_list.txt"
    annotaions_filepath_list = read_text_lines(annotation_list_path)
    
    
    
    

    for filename in tqdm(annotaions_filepath_list):
        
        filewise_info = dict()
        
        filename_id = int(os.path.basename(filename)[:-5])
        
        saved_json_path = os.path.join(root_path,filename.replace("annotations","annotations_simple"))
        os.makedirs(os.path.dirname(saved_json_path),exist_ok=True)
        
        
        annotation_path = filename
        right_annotation_path = annotation_path.replace("image_00","image_01")
        left_image_path = filename.replace("annotations","data_2d_raw").replace(".json",".png")
        right_image_path = left_image_path.replace("image_00","image_01")
        left_depth_monodepthv2_path = filename.replace("annotations",'monocular_depth/monodepthV2/data_2d_raw').replace(".json",'.png')
        right_depth_monodepthv2_path = left_depth_monodepthv2_path.replace("image_00","image_01")
        left_depth_monodepth_metricv2_path = filename.replace("annotations",'monocular_depth/Metric3DV2/data_2d_raw').replace(".json",'.png')
        left_depth_monodepth_metricv2_path  = left_depth_monodepth_metricv2_path.replace(".png","_dpt.png")
        right_depth_monodepth_metricv2_path = left_depth_monodepthv2_path.replace("image_00","image_01")
        left_depth_sparse_gt_path =filename.replace("annotations",'projected_sparse_lidar/data_2d_raw').replace(".json",'.png')
        right_depth_sparse_gt_path = left_depth_sparse_gt_path.replace("image_00","image_01")
        left_depth_stereo_path =filename.replace("annotations","PseudoDepth_NMRFStereo/data_2d_raw").replace(".json",".png")
        right_depth_stereo_path = left_depth_stereo_path.replace("image_00","image_01")
        annotation_path_abs = os.path.join(root_path, annotation_path)
        left_image_path_abs = os.path.join(root_path, left_image_path)
        right_image_path_abs = os.path.join(root_path, right_image_path)
        left_depth_monodepthv2_path_abs = os.path.join(root_path, left_depth_monodepthv2_path)
        right_depth_monodepthv2_path_abs = os.path.join(root_path, right_depth_monodepthv2_path)
        left_depth_monodepth_metricv2_path_abs = os.path.join(root_path, left_depth_monodepth_metricv2_path)
        right_depth_monodepth_metricv2_path_abs = os.path.join(root_path, right_depth_monodepth_metricv2_path)
        left_depth_sparse_gt_path_abs = os.path.join(root_path, left_depth_sparse_gt_path)
        right_depth_sparse_gt_path_abs = os.path.join(root_path, right_depth_sparse_gt_path)
        left_depth_stereo_path_abs = os.path.join(root_path, left_depth_stereo_path)
        right_depth_stereo_path_abs = os.path.join(root_path, right_depth_stereo_path)
        
        right_annotation_path_abs = os.path.join(root_path,right_annotation_path)
                
        
        assert os.path.exists(left_depth_sparse_gt_path_abs)
        assert os.path.exists(right_depth_sparse_gt_path_abs)

        assert os.path.exists(left_depth_monodepth_metricv2_path_abs)
        assert os.path.exists(right_depth_monodepth_metricv2_path_abs)

        assert os.path.exists(left_depth_monodepthv2_path_abs)
        assert os.path.exists(right_depth_monodepthv2_path_abs)

        assert os.path.exists(left_image_path_abs)
        assert os.path.exists(right_image_path_abs)

        assert os.path.exists(annotation_path_abs)

        assert os.path.exists(left_depth_stereo_path_abs)
        assert os.path.exists(right_depth_stereo_path_abs)
        assert os.path.exists(right_annotation_path_abs)
        
        

        left_cam_to_lidar_pose = np.array([[ 4.36121151e-02, -9.12146196e-02,  9.94875611e-01,  8.04391442e-01],
                                            [-9.99038885e-01,  5.08141636e-04,  4.38409241e-02,  2.99348957e-01],
                                            [-4.50416730e-03, -9.95831360e-01, -9.11039874e-02, -1.77022582e-01],
                                            [ 0.00000000e+00 , 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

        right_cam_to_lidar_pose = np.array([[ 4.36114960e-02, -9.12138106e-02,  9.94875936e-01,  8.30304892e-01],
                                            [-9.99038705e-01,  5.07456168e-04,  4.38407394e-02, -2.94263375e-01],
                                            [-4.50373702e-03, -9.95830873e-01, -9.11044252e-02,-1.79698816e-01],
                                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        
        
        
        raw_ck0,world_2_left_cam = read_annotation(annotation_filename=annotation_path_abs)
        raw_ck1,world_2_right_cam = read_annotation(annotation_filename=right_annotation_path_abs)
        
        left_cam_2_world = torch.linalg.inv(world_2_left_cam).cpu().numpy()
        right_cam_2_world = torch.linalg.inv(world_2_right_cam).cpu().numpy()
        raw_ck0 = raw_ck0.cpu().numpy()
        
        
        filewise_info['id'] = filename_id
        
        filewise_info['left_image_path'] = left_image_path
        filewise_info['right_image_path'] = right_image_path
        filewise_info['left_image_anno'] = annotation_path
        filewise_info['right_image_anno'] = right_annotation_path
        
        filewise_info['left_image_pseudo_depth']= dict()
        filewise_info['left_image_pseudo_depth']['depthanythingv2'] = left_depth_monodepthv2_path
        filewise_info['left_image_pseudo_depth']['metricv2'] = left_depth_monodepth_metricv2_path
        filewise_info['left_image_pseudo_depth']['stereo'] = left_depth_stereo_path
        filewise_info['left_image_pseudo_depth']['lidar'] = left_depth_sparse_gt_path

        filewise_info['right_image_pseudo_depth']= dict()
        filewise_info['right_image_pseudo_depth']['depthanythingv2'] = right_depth_monodepthv2_path
        filewise_info['right_image_pseudo_depth']['metricv2'] = right_depth_monodepth_metricv2_path
        filewise_info['right_image_pseudo_depth']['stereo'] =  right_depth_stereo_path
        filewise_info['right_image_pseudo_depth']['lidar'] = right_depth_sparse_gt_path
        
        
        filewise_info['raw_ck'] = raw_ck0.tolist()
        filewise_info['left_cam_to_lidar'] = left_cam_to_lidar_pose.tolist()
        filewise_info['right_cam_to_lidar'] = right_cam_to_lidar_pose.tolist()
        
        filewise_info['left_cam_to_world'] = left_cam_2_world.tolist()
        filewise_info['right_cam_to_world'] = right_cam_2_world.tolist()
        
        
        save_dict_to_json(data_dict=filewise_info,
                          filename=saved_json_path)
        
        
        
        
        
        
        
        
        
        
        
