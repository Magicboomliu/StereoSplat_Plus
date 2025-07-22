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
import open3d as o3d

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



import pickle


def create_camera_frustum_mesh(scale=0.2, color=[1.0, 0.1, 0.0]):
    vertices = np.array([
        [0, 0, 0],       # Camera origin
        [1, 1, 2],
        [-1, 1, 2],
        [-1, -1, 2],
        [1, -1, 2],
    ]) * scale

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return line_set

def compute_cumulative_lengths(positions):
    dists = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    return np.concatenate([[0], np.cumsum(dists)])

def create_bin_obb(p1, p2, width=2.0, height=2.0):
    center = (p1 + p2) / 2
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-5:
        return None
    z = direction / length
    up = np.array([0, 0, 1]) if abs(z[2]) < 0.9 else np.array([0, 1, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    extent = np.array([width, height, length])
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    obb.color = [0.0, 1.0, 0.0]
    return obb

def create_start_marker(p1, p2, width=2.0, height=2.0, thickness=0.01):
    """
    创建实体矩形块作为 bin 起点标记（全色实心）
    """
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-5:
        return None

    z = direction / length
    up = np.array([0, 0, 1]) if abs(z[2]) < 0.9 else np.array([0, 1, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)

    # 创建 box：初始在原点，中心是 box 的角点
    box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=thickness)
    box.paint_uniform_color([1.0, 0.0, 0.0])  # 绿色

    # 平移 box 到中心（从 corner 平移到中心）
    center_offset = np.array([width / 2, height / 2, thickness / 2])
    box.translate(-center_offset)

    # 变换方向并移动到 p1
    box.rotate(R, center=(0, 0, 0))
    box.translate(p1)

    return box

def draw_all_cameras_with_bins(c2w_matrices, scale=0.1, axis_size=100, step=10.0, overlap=3.0, bin_width=2.0, bin_height=2.0,
                               vis=True):
    geometries = []

    # 左相机轨迹中心用于 bin 划分
    c2w_matrices_left = c2w_matrices[0::2]
    centers = c2w_matrices_left[:, :3, 3]
    center_mean = centers.mean(axis=0)

    # 添加全局坐标轴
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
    axis.translate(center_mean)
    geometries.append(axis)

    # 所有相机视锥体（左右交替）
    for i in range(c2w_matrices.shape[0]):
        cam_mesh = create_camera_frustum_mesh(scale=scale)
        cam_mesh.transform(c2w_matrices[i])
        geometries.append(cam_mesh)

    # 累积路径长度，用于 bin 计算
    cum_dists = compute_cumulative_lengths(centers)
    total_len = cum_dists[-1]
    bin_start = 0.0
    bin_len = step + overlap

    red_marker_frame_idxs = []  # 新增：记录所有红色 marker 对应的 frame index
    
    while bin_start < total_len:
        bin_end = bin_start + bin_len
        idx = np.where((cum_dists >= bin_start) & (cum_dists <= bin_end))[0]
        if len(idx) > 1:
            p1, p2 = centers[idx[0]], centers[idx[-1]]
            obb = create_bin_obb(p1, p2, width=bin_width, height=bin_height)
            marker = create_start_marker(p1, p2, width=bin_width, height=bin_height)
            if obb:
                geometries.append(obb)
            if marker:
                geometries.append(marker)
                red_marker_frame_idxs.append(idx[0])  # ✅ 添加红色 marker 的起始帧 idx
        bin_start += step

    # 可视化所有几何体
    if vis:
        o3d.visualization.draw_geometries(geometries)

    return red_marker_frame_idxs

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def save_pickle(filename,data):
    """
    保存任意 Python 对象为 .pkl 文件

    Args:
        data: 要保存的数据（可以是 dict, list, numpy array 等）
        filename (str): 输出文件路径，应以 .pkl 结尾
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


if __name__=="__main__":
    
    root_path = "/data1/StereoDatasets/KITTI/KITTI360/"
    annotation_list_path = "/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/avaliable_lists/2013_05_28_drive_0000_sync_list.txt"
    
    output_folder = "/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/semi_global_maps/2013_05_28_drive_0000_sync"
    os.makedirs(output_folder,exist_ok=True)
    annotaions_filepath_list = sorted(read_text_lines(annotation_list_path))
    
    
    left_cam_2_world_list = []
    
    right_cam_2_world_list = []

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
        
        
        left_cam_2_world_list.append(left_cam_2_world.tolist())
        right_cam_2_world_list.append(right_cam_2_world.tolist())
        
        
    c2w_matrices_left = np.array(left_cam_2_world_list)
    c2w_matrices_right = np.array(right_cam_2_world_list)


    N = c2w_matrices_left.shape[0]
    merged = np.empty((2 * N, 4, 4), dtype=c2w_matrices_left.dtype)
    # 交替填入 A 和 B
    merged[0::2] =  c2w_matrices_left # 偶数索引放 A[0], A[1], ...
    merged[1::2] = c2w_matrices_right  # 奇数索引放 B[0], B[1], ...
    

    input_idx_list = draw_all_cameras_with_bins(c2w_matrices=merged,
                                scale=0.1,axis_size=10,step=10,
                                overlap=5,bin_width=4,bin_height=4,
                                vis=False)
    
    input_frame_list = []
    key_frame_nums_inside_one_semi_global_map = 40

    for idx, input_id in enumerate(input_idx_list):
        input_frame_list.append(annotaions_filepath_list[input_id])
    
    saved_dict_info_global_info = dict(
        key_frames_list = input_frame_list,
        all_frames_list = annotaions_filepath_list
        
    )
    save_pickle("{}/global.pkl".format(output_folder),
                saved_dict_info_global_info)

    current_key_frame_numbers_stack =[]
    semi_global_frames_stack = []
    semi_global_map_id = 0
    
    
    counter = 0
    for idx,fname in enumerate(annotaions_filepath_list):
        
        semi_global_frames_stack.append(fname)
        
        if fname in input_frame_list:
            current_key_frame_numbers_stack.append(fname)

        
        if len(current_key_frame_numbers_stack)%key_frame_nums_inside_one_semi_global_map==0 and len(current_key_frame_numbers_stack)>0:
            counter = counter +1
            saved_dict_info = dict(
                key_frames_list = current_key_frame_numbers_stack,
                all_frames_list = semi_global_frames_stack
            )
            save_pickle("{}/semi_global_{}.pkl".format(output_folder,semi_global_map_id),
                        saved_dict_info)
            
            current_key_frame_numbers_stack = []
            semi_global_frames_stack = []
            semi_global_map_id+=1
    
    print("Done!")
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
