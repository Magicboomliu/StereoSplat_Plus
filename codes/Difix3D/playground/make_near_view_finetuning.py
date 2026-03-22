import json
import os
import numpy as np 
import skimage
import skimage.io
import argparse
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm


def write_pose_to_json(pose, json_path: str) -> None:
    """
    Save a 4x4 camera pose to a JSON file.

    Args:
        pose: 4x4 pose matrix
        json_path: output JSON file path
    """
    pose = np.asarray(pose, dtype=float)

    if pose.shape != (4, 4):
        raise ValueError(f"Expected pose shape (4, 4), but got {pose.shape}")

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "pose": pose.tolist()
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def read_pose_from_json(json_path: str) -> np.ndarray:
    """
    Read a 4x4 camera pose from a JSON file.

    Args:
        json_path: input JSON file path

    Returns:
        pose: numpy array of shape (4, 4)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pose = np.asarray(data["pose"], dtype=float)

    if pose.shape != (4, 4):
        raise ValueError(f"Expected pose shape (4, 4), but got {pose.shape}")

    return pose

def compute_psnr_ssim_from_paths(img1_path: str, img2_path: str):
    """
    Compute PSNR and SSIM from two image paths.

    Args:
        img1_path: path to the first image
        img2_path: path to the second image

    Returns:
        psnr: float
        ssim: float
    """
    img1 = np.array(Image.open(img1_path).convert("RGB"), dtype=np.float32) / 255.0
    img2 = np.array(Image.open(img2_path).convert("RGB"), dtype=np.float32) / 255.0

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes do not match: {img1.shape} vs {img2.shape}")

    psnr = peak_signal_noise_ratio(img1, img2, data_range=1.0)
    ssim = structural_similarity(img1, img2, channel_axis=2, data_range=1.0)

    return float(psnr), float(ssim)


def save_dict_to_json(data: dict, json_path: str) -> None:
    """将字典写入 JSON；递归把 numpy 数组/标量转为 Python 原生类型。"""

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(x) for x in obj]
        return obj

    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=4, ensure_ascii=False)


def making_near_view_finetuning_dataset(args):
    
    
    all_results_dict = {}
    
    # creating the difix dataset.
    training_dict = {}
    global_training_index = 0
    
    root_folder = args.root_folder
    
    difix_root_folder = os.path.join(root_folder, args.dataset_type)
    assert os.path.exists(difix_root_folder)
    
    output_filename_folder = os.path.join(args.output_filename_folder, args.dataset_type)
    os.makedirs(output_filename_folder, exist_ok=True)
    
    save_json_filename_path = os.path.join(output_filename_folder, "all_results_dict.json")
    

    for scene_name in tqdm(sorted(os.listdir(difix_root_folder))):
        
        scene_folder = os.path.join(difix_root_folder, scene_name)
        assert os.path.exists(scene_folder)
        raw_image_folder = os.path.join(scene_folder, "raw_images")
        gt_image_folder = os.path.join(scene_folder, "gt_images")
        ref_image_folder = os.path.join(scene_folder, "ref_images")
        relative_pose = os.path.join(scene_folder, scene_name,"relative_poses")
        
        for image_id  in range(len(os.listdir(raw_image_folder))):
            
            first_frame_left_id = len(os.listdir(raw_image_folder)) - 2
            first_frame_right_id = len(os.listdir(raw_image_folder)) - 1
            
            if image_id == first_frame_left_id or image_id == first_frame_right_id:
                continue
            
            current_raw_image_path = os.path.join(raw_image_folder, "rendered_{}.png".format(image_id))
            current_gt_image_path = os.path.join(gt_image_folder, "gt_{}.png".format(image_id))
            current_ref_image_path = os.path.join(ref_image_folder, "reference_{}.png".format(image_id))
            current_relative_pose_path = os.path.join(relative_pose, "relative_pose_{}.txt".format(image_id))
            
            
            assert os.path.exists(current_raw_image_path)
            assert os.path.exists(current_gt_image_path)
            assert os.path.exists(current_ref_image_path)
            assert os.path.exists(current_relative_pose_path)
            
            
            current_psnr, current_ssim = compute_psnr_ssim_from_paths(current_raw_image_path, 
                                                                      current_gt_image_path)
            
            
            if current_psnr >=args.psnr_threshold:
                data_item_dict = {}
                data_item_dict["image"] = current_raw_image_path
                data_item_dict["target_image"] = current_gt_image_path
                data_item_dict["ref_image"] = current_ref_image_path
                data_item_dict["prompt"] = "remove degradation"
                
                data_item_dict['relative_pose'] = read_pose_from_json(current_relative_pose_path)
            
                training_dict[global_training_index] = data_item_dict
                global_training_index += 1

    # 从 training 中按顺序每 10 条取 1 条作为 validation，其余重新连续编号为训练集
    val_dict = {}
    new_training_dict = {}
    new_train_idx = 0
    for i in range(global_training_index):
        item = training_dict[i]
        if i % 10 == 0:
            val_dict[len(val_dict)] = item
        else:
            new_training_dict[new_train_idx] = item
            new_train_idx += 1
    training_dict = new_training_dict
    
    
    all_results_dict["training"] = new_training_dict
    all_results_dict["validation"] = val_dict
    
    save_dict_to_json(all_results_dict, 
                      save_json_filename_path)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--root_folder", type=str, default="/data1/zliu/IROS26/Difix3D_Pose_Prompt/")
    parser.add_argument("--dataset_type", type=str, default="Validation_Set")
    parser.add_argument("--output_filename_folder", type=str, default="/home/zliu/IROS2026/StereoSplat_Plus/codes/Difix3D/filenames/")
    parser.add_argument("--psnr_threshold", type=float, default=20.0)
    args = parser.parse_args()
    root_folder = args.root_folder
    
    making_near_view_finetuning_dataset(args)