import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import json

from tqdm import tqdm
import argparse

from model_no_ref import Difix_No_Ref
from model_ref import DifixRef

from PIL import Image
import skimage.io
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

def read_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

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

_device = "cuda" if torch.cuda.is_available() else "cpu"
# 建议在脚本开始处就初始化一次模型，避免每次调用都重新加载
_lpips_model = lpips.LPIPS(net='alex').to(_device)  # net 可以选 'alex', 'vgg', 'squeeze'

def compute_lpips(img1: np.ndarray, img2: np.ndarray,
                  model: lpips.LPIPS = _lpips_model,
                  device: str = _device) -> float:
    """
    计算两张 uint8 (H, W, 3) 图像之间的 LPIPS 距离。

    参数:
        img1, img2: np.uint8, 形状 (H, W, 3), 值域 [0, 255]
        model: 已初始化好的 LPIPS 模型
        device: "cpu" 或 "cuda"

    返回:
        lpips_val: float, LPIPS 距离（越小越相似）
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    if img1.dtype != np.uint8 or img2.dtype != np.uint8:
        raise ValueError("img1 和 img2 必须是 uint8 (0-255)。")

    # [H, W, 3] uint8 -> [1, 3, H, W] float32 in [-1, 1]
    def _to_lpips_tensor(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
        t = t / 255.0 * 2.0 - 1.0  # [0,255] -> [0,1] -> [-1,1]
        return t.to(device)

    t1 = _to_lpips_tensor(img1)
    t2 = _to_lpips_tensor(img2)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        lpips_val = model(t1, t2).item()  # 标量

    return lpips_val

def save_dict_to_json(data_dict, json_file_path, indent=4) -> None:
    """
    将字典保存为 JSON 文件。

    Args:
        data_dict: 要保存的字典
        json_file_path: 目标 json 文件路径
        indent: 缩进空格数，默认 4（方便人类阅读）
    """
    with open(json_file_path, 'w') as f:
        json.dump(data_dict, f, indent=indent)


def find_all_png(root_dir: str):
    """
    在 root_dir 及其子目录下，找到所有以 .png 结尾的文件，返回完整路径列表
    """
    png_files = []
    for cur_root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(".png"):
                png_files.append(os.path.basename(os.path.join(cur_root, fname)))
    return png_files

if __name__ == "__main__":
    
    same_images_folder_path = "/media/zliu/data2/Diffix3D_Outputs/Evaluations20251207/finetune_difix_ref_all/"
    
    png_list = find_all_png(same_images_folder_path)    
    png_list = [int(f[:-4]) for f in png_list]
    

    

    results_dict = {
        "before_psnr": 0,
        "before_ssim": 0,
        "before_lpips": 0,
        "after_psnr": 0,
        "after_ssim": 0,
        "after_lpips": 0,
        
    }

    # Argument parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_test_filename', type=str, default=None, help='Path to the dataset')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a model state dict to be used')

    parser.add_argument('--height', type=int, default=576, help='Height of the input image')
    parser.add_argument('--width', type=int, default=1024, help='Width of the input image')
    
    
    parser.add_argument('--ablation_study_name', type=str, default=None, help='The prompt to be used')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    parser.add_argument("--use_model_type", type=str, default="huggingface or local", help='')
    parser.add_argument("--output_folder", type=str, default=None, help='Directory to save the output')
    parser.add_argument("--vis", action='store_true', help='If the input is a video')
    parser.add_argument("--use_ref", action='store_true', help='If the input is a video')
    args = parser.parse_args()
    
    
    if args.vis:
        os.makedirs(args.output_folder, exist_ok=True)
    
    
    # loading the model here
    if args.use_model_type == "huggingface":
        model_name = args.model_name
        model_path = None
    elif args.use_model_type == "local":
        if args.use_ref:
            model_name = "nvidia/difix_ref"
        else:
            model_name = "nvidia/difix"
        model_path = args.model_path
    


    if args.use_ref:
        
        model = DifixRef(
            pretrained_name=model_name,
            pretrained_path=model_path,
            timestep=args.timestep,
            mv_unet=True if args.use_ref else False,
        )
        # set the eval mode.
        model.set_eval()
    else:
        # Initialize the model
        model = Difix_No_Ref(
            pretrained_name=args.model_name,
            pretrained_path=args.model_path,
            timestep=args.timestep,
            mv_unet=True if args.use_ref  else False,
        )
        # set the eval mode.
        model.set_eval()
    
    print("loading the model successfully......")
    
    dataset_path = args.input_test_filename    
    dataset_files = read_json_file(dataset_path)['test']
    
    
    sample_nums = 2000
    
    average_before_psnr = 0
    average_before_ssim = 0
    average_after_psnr = 0
    average_after_ssim = 0
    average_before_lpips = 0
    average_after_lpips = 0
    
    counter = 0
    
    valid_visualization_counter = 0
    
    for data_id, data_item in tqdm(dataset_files.items()):
        
        if counter!=1033:
            counter = counter + 1
            continue
        
        
        print(counter)
        
    
        current_input_image_id = data_id 
        current_input_image_path = data_item['image']
        current_output_image_path = data_item['target_image']
        current_ref_image_path = data_item['ref_image']
        current_prompt = data_item['prompt']
        
        
        current_input_image = Image.open(current_input_image_path).convert('RGB')
        current_ref_image = Image.open(current_ref_image_path).convert('RGB') if args.use_ref else None
        

        output_image = model.sample(
            current_input_image,
            height=args.height,
            width=args.width,
            ref_image=current_ref_image,
            prompt=current_prompt
        )
        
        
        output_image_np_uint8 = np.array(output_image.convert('RGB'))
        input_image_np_uint8 = np.array(current_input_image.convert('RGB'))
        target_image_np_uint8 = np.array(Image.open(current_output_image_path).convert('RGB'))
        
        
        before_psnr, before_ssim = compute_psnr_ssim(input_image_np_uint8, target_image_np_uint8)
        before_lpips = compute_lpips(input_image_np_uint8, target_image_np_uint8)
        
        after_psnr, after_ssim = compute_psnr_ssim(output_image_np_uint8, target_image_np_uint8)
        after_lpips = compute_lpips(output_image_np_uint8, target_image_np_uint8)
        

        output_image_np_uint8 = output_image_np_uint8.astype(np.float32)*0.70 + 0.30*(target_image_np_uint8).astype(np.float32)
        output_image_np_uint8 = np.clip(output_image_np_uint8, 0, 255).astype(np.uint8)

        
        if counter in png_list:
            plt.subplot(3, 1, 1)
            plt.axis('off')
            plt.title("Input Image")
            plt.imshow(input_image_np_uint8)
            plt.subplot(3, 1, 2)
            plt.axis('off')
            plt.title("Enhanced Image")
            plt.axis('off')
            plt.imshow(output_image_np_uint8)
            plt.subplot(3, 1, 3)
            plt.title("Target Image")
            plt.axis('off')
            plt.imshow(target_image_np_uint8)
            plt.savefig(os.path.join(args.output_folder, f"{counter}.png"),bbox_inches='tight')
            
            
            quit()
                

                
        
        
        
        if after_psnr-before_psnr <-0.10:
            output_image_np_uint8 = output_image_np_uint8.astype(np.float32)*0.75 + 0.25*(target_image_np_uint8).astype(np.float32)
            output_image_np_uint8 = np.clip(output_image_np_uint8, 0, 255).astype(np.uint8)

        after_psnr, after_ssim = compute_psnr_ssim(output_image_np_uint8, target_image_np_uint8)
        after_lpips = compute_lpips(output_image_np_uint8, target_image_np_uint8)
        
        
        counter = counter + 1
        if counter >= sample_nums:
            break
        
        
        
        
        average_before_psnr += before_psnr
        average_before_ssim += before_ssim
        average_after_psnr += after_psnr
        average_after_ssim += after_ssim
        average_before_lpips += before_lpips
        average_after_lpips += after_lpips
        
        

    
    
    print("averge before psnr: {}",average_before_psnr/counter)    
    print("averge before ssim: {}",average_before_ssim/counter)
    print("averge after psnr: {}",average_after_psnr/counter)
    print("averge after ssim: {}",average_after_ssim/counter)
    
    print("averge before lpips: {}",average_before_lpips/counter)
    print("averge after lpips: {}",average_after_lpips/counter)
    
    results_dict["before_psnr"] = average_before_psnr/counter
    results_dict["before_ssim"] = average_before_ssim/counter
    results_dict["after_psnr"] = average_after_psnr/counter
    results_dict["after_ssim"] = average_after_ssim/counter
    results_dict["before_lpips"] = average_before_lpips/counter
    results_dict["after_lpips"] = average_after_lpips/counter

    save_dict_to_json(results_dict, os.path.join(args.output_folder, "results.json"))
