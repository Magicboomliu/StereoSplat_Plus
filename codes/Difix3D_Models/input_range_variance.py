import os
import numpy as np
import torch
import json
import skimage.io
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def read_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json_file(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def compute_psnr_ssim(img1: np.ndarray, img2: np.ndarray):
    """
    img1, img2: uint8, shape (H, W, 3), range [0, 255]
    return: psnr(float), ssim(float)
    """
    # 基本检查
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    if img1.dtype != np.uint8 or img2.dtype != np.uint8:
        raise ValueError("Both images must be uint8 (0-255).")

    # 转成 float 计算
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    # PSNR
    psnr_val = peak_signal_noise_ratio(img1_f, img2_f, data_range=255.0)

    # SSIM（新版本 skimage 用 channel_axis，旧版本用 multichannel）
    try:
        ssim_val = structural_similarity(
            img1_f, img2_f, data_range=255.0, channel_axis=-1
        )
    except TypeError:
        # 兼容旧版 skimage
        ssim_val = structural_similarity(
            img1_f, img2_f, data_range=255.0, multichannel=True
        )

    return psnr_val, ssim_val





if __name__ == "__main__":
    
    # 存储所有的 PSNR 和 SSIM 值
    psnr_values = []
    ssim_values = []
    # 存储每个样本的详细信息
    sample_info = []
    
    input_json_path = "/home/zliu/Desktop/TEMP/FeedStereoGS/filenames/kitti360/difix_dataset/trainval.json"
    
    training_readed_data = read_json_file(input_json_path)['train']
    
    
    
    
    all_counters_nums = 50000
    
    counter = 0
    
    for key in tqdm(training_readed_data.keys()):
        current_data = training_readed_data[key]
        current_input_image_path = current_data['image']
        current_target_image_path = current_data['target_image']
        current_ref_image_path = current_data['ref_image']
        current_prompt = current_data['prompt']
        
        current_input_image = Image.open(current_input_image_path)
        current_target_image = Image.open(current_target_image_path)
        
        
        current_input_image_np = np.array(current_input_image)
        current_target_image_np = np.array(current_target_image)
        
        
        computed_psnr, computed_ssim = compute_psnr_ssim(current_input_image_np, current_target_image_np)
        
        # 收集所有值
        psnr_values.append(computed_psnr)
        ssim_values.append(computed_ssim)
        
        # 存储样本信息
        sample_info.append({
            'key': key,
            'psnr': computed_psnr,
            'ssim': computed_ssim,
            'input_path': current_input_image_path,
            'target_path': current_target_image_path
        })
        
        counter = counter + 1
        
        if counter >= all_counters_nums:
            break
    
    # 计算统计信息
    psnr_array = np.array(psnr_values)
    ssim_array = np.array(ssim_values)
    
    highest_psnr = float(np.max(psnr_array))
    lowest_psnr = float(np.min(psnr_array))
    std_psnr = float(np.std(psnr_array))
    mean_psnr = float(np.mean(psnr_array))
    
    highest_ssim = float(np.max(ssim_array))
    lowest_ssim = float(np.min(ssim_array))
    std_ssim = float(np.std(ssim_array))
    mean_ssim = float(np.mean(ssim_array))
    
    # 构建结果字典
    results = {
        "psnr": {
            "max": highest_psnr,
            "min": lowest_psnr,
            "mean": mean_psnr,
            "std": std_psnr
        },
        "ssim": {
            "max": highest_ssim,
            "min": lowest_ssim,
            "mean": mean_ssim,
            "std": std_ssim
        },
        "total_samples": len(psnr_values)
    }
    
    # 保存为 JSON
    output_json_path = "/home/zliu/Desktop/TEMP/FeedStereoGS/codes/Difix3D/src/input_range_variance_stats.json"
    save_json_file(results, output_json_path)
    
    print(f"\n统计结果已保存到: {output_json_path}")
    print(f"PSNR - 最大值: {highest_psnr:.4f}, 最小值: {lowest_psnr:.4f}, 均值: {mean_psnr:.4f}, 标准差: {std_psnr:.4f}")
    print(f"SSIM - 最大值: {highest_ssim:.4f}, 最小值: {lowest_ssim:.4f}, 均值: {mean_ssim:.4f}, 标准差: {std_ssim:.4f}")
    
    # 找到PSNR最高和最低的10个样本
    sample_info_sorted = sorted(sample_info, key=lambda x: x['psnr'], reverse=True)
    top_10_high_psnr = sample_info_sorted[:10]
    top_10_low_psnr = sample_info_sorted[-10:]
    
    # 创建输出目录
    output_dir = "/home/zliu/Desktop/TEMP/FeedStereoGS/codes/Difix3D/src/psnr_examples"
    os.makedirs(output_dir, exist_ok=True)
    high_psnr_dir = os.path.join(output_dir, "high_psnr")
    low_psnr_dir = os.path.join(output_dir, "low_psnr")
    os.makedirs(high_psnr_dir, exist_ok=True)
    os.makedirs(low_psnr_dir, exist_ok=True)
    
    # 保存PSNR最高的10个样本
    print(f"\n保存PSNR最高的10个样本...")
    for idx, sample in enumerate(tqdm(top_10_high_psnr, desc="High PSNR")):
        input_img = Image.open(sample['input_path'])
        target_img = Image.open(sample['target_path'])
        
        # 确保两个图像尺寸相同
        if input_img.size != target_img.size:
            target_img = target_img.resize(input_img.size, Image.Resampling.LANCZOS)
        
        # 水平拼接 (H方向，即宽度方向)
        concat_img = Image.new('RGB', (input_img.width + target_img.width, input_img.height))
        concat_img.paste(input_img, (0, 0))
        concat_img.paste(target_img, (input_img.width, 0))
        
        # 保存拼接后的图像
        output_path = os.path.join(high_psnr_dir, f"high_psnr_{idx+1:02d}_psnr_{sample['psnr']:.2f}.png")
        concat_img.save(output_path)
    
    # 保存PSNR最低的10个样本
    print(f"\n保存PSNR最低的10个样本...")
    for idx, sample in enumerate(tqdm(top_10_low_psnr, desc="Low PSNR")):
        input_img = Image.open(sample['input_path'])
        target_img = Image.open(sample['target_path'])
        
        # 确保两个图像尺寸相同
        if input_img.size != target_img.size:
            target_img = target_img.resize(input_img.size, Image.Resampling.LANCZOS)
        
        # 水平拼接 (H方向，即宽度方向)
        concat_img = Image.new('RGB', (input_img.width + target_img.width, input_img.height))
        concat_img.paste(input_img, (0, 0))
        concat_img.paste(target_img, (input_img.width, 0))
        
        # 保存拼接后的图像
        output_path = os.path.join(low_psnr_dir, f"low_psnr_{idx+1:02d}_psnr_{sample['psnr']:.2f}.png")
        concat_img.save(output_path)
    
    print(f"\n所有示例图像已保存到: {output_dir}")
    print(f"  - 高PSNR样本: {high_psnr_dir}")
    print(f"  - 低PSNR样本: {low_psnr_dir}")
        



