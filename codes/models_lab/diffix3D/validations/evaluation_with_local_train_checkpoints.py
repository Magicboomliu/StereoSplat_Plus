import os
import imageio
import argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import cv2
import sys
sys.path.append("..")
from model import Difix

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import lpips
import json
import skimage.io




def compute_psnr_ssim(target: np.ndarray, reference: np.ndarray):
    """
    target, reference: np.uint8 arrays of shape [H, W, 3] in RGB
    returns: (psnr, ssim)
    """
    if target.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {target.shape} vs {reference.shape}")
    if target.dtype != np.uint8 or reference.dtype != np.uint8:
        raise TypeError("Inputs must be uint8 images.")

    # data_range is 255 for 8-bit images
    psnr = peak_signal_noise_ratio(reference, target, data_range=255)
    # For skimage >=0.19 use channel_axis=-1 (older versions: multichannel=True)
    ssim = structural_similarity(reference, target, data_range=255, channel_axis=-1)
    return psnr, ssim

def load_image_local(image_path):
    image = np.array(Image.open(image_path))
    return image

def compute_lpips(target_uint8: np.ndarray, ref_uint8: np.ndarray, device: str = None) -> float:
    """
    target_uint8, ref_uint8: np.uint8 images, shape [H,W,3], RGB
    returns: LPIPS distance as float (lower is better)
    """
    if target_uint8.shape != ref_uint8.shape:
        raise ValueError(f"Images must have the same shape, got {target_uint8.shape} vs {ref_uint8.shape}")
    if target_uint8.dtype != np.uint8 or ref_uint8.dtype != np.uint8:
        raise TypeError("Inputs must be uint8 images.")

    # Choose device
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _lpips_net.to(dev)

    # to torch, NCHW in [-1, 1]
    def to_tensor(img_u8: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img_u8).float() / 255.0              # [H,W,3] in [0,1]
        t = t.permute(2, 0, 1).unsqueeze(0)                       # [1,3,H,W]
        t = t * 2 - 1                                             # to [-1,1]
        return t.to(dev)

    with torch.no_grad():
        d = _lpips_net(to_tensor(target_uint8), to_tensor(ref_uint8))  # [1,1] tensor
    return float(d.item())

def save_dict_to_json(data: dict, filename: str) -> None:
    """
    Save a Python dictionary to a JSON file.

    Args:
        data (dict): Dictionary to save.
        filename (str): Path to the output JSON file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_validation_filenames(filename: str) -> dict:
    """
    Read a JSON file and return it as a Python dictionary.

    Args:
        filename (str): Path to the input JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    testing_data = data['test']
    
    return data

def get_subset_sign(validation_content,sub_folder_list):
    input_image_path = validation_content['image']
    for sub_folder_name in sub_folder_list:
        if sub_folder_name in input_image_path:
            return sub_folder_name
        
def stack_images_vertically(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray) -> np.ndarray:
    """
    Stack 3 RGB images of shape (H, W, 3) vertically -> (3H, W, 3).

    Args:
        img1, img2, img3: np.ndarray, dtype=uint8, shape=(H,W,3)

    Returns:
        np.ndarray of shape (3H, W, 3), dtype=uint8
    """
    # 检查形状一致
    if not (img1.shape == img2.shape == img3.shape):
        raise ValueError(f"All images must have the same shape, got {img1.shape}, {img2.shape}, {img3.shape}")
    if img1.dtype != np.uint8 or img2.dtype != np.uint8 or img3.dtype != np.uint8:
        raise TypeError("All images must be uint8 arrays.")

    return np.vstack([img1, img2, img3])    
    
def PIL_to_np(image):
    return np.array(image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,default="/data3/zliu/CVPR25/GSEnhanceDataset/KITTI360/")
    parser.add_argument("--validation_filename_path", type=str, required=True,default="/home/zliu/Project2025/Difix3D/FeedStereoGS/filenames/KITTI360/train_val_data_split.json")
    parser.add_argument('--prompt', type=str, required=True, help='The prompt to be used')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    parser.add_argument("--ablation_type", type=str, required=True,default="official_diffix3D")
    parser.add_argument("--no_ref", action="store_true")
    parser.add_argument("--vis",action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Argument parser
    args = parse_args()
    sub_folder_list = os.listdir(args.root_dir)

    # Create the metric network once (alex = default/best-known tradeoff)
    _lpips_net = lpips.LPIPS(net='alex')  # options: 'alex', 'vgg', 'squeeze'
    _lpips_net.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # saved visualization path
    saved_visualization_path = os.path.join(args.output_dir, "visualizations",args.ablation_type)
    os.makedirs(saved_visualization_path, exist_ok=True)

    all_metrics_dict = {
        key:{"psnr":0,"ssim":0,"lpips":0} for key in sub_folder_list
    }
    
    all_metrics_dict_counter ={
        key:0 for key in sub_folder_list
    }
    
    all_metrics_dict.update({"overall":{"psnr":0,"ssim":0,"lpips":0}})
    
    # total folder indictor 
    total_folder_indicator = 0
    validation_dict_contents = read_validation_filenames(args.validation_filename_path)
    validation_dict_contents = validation_dict_contents['test']

    # state the model name and the model path for loading the model.
    # Initialize the model
    assert os.path.exists(args.model_path), "Model path does not exist"
    
    
    if args.no_ref:
        mv_unet = False
    else:
        mv_unet = True
    
    model = Difix(
        pretrained_name=args.model_name,
        pretrained_path=args.model_path,
        timestep=args.timestep,
        mv_unet=mv_unet)
    
    model.set_eval()    
    print("Successfully loading the pre-trained model.....")
    
    
    for validation_key,validation_content in tqdm(validation_dict_contents.items(), desc="Evaluating"):
        
        current_subset_sign = get_subset_sign(validation_content,sub_folder_list)

        input_image_path = validation_content['image']
        target_image_path = validation_content['target_image']
        ref_image_path = validation_content['ref_image']
        prompt = validation_content['prompt']
        saved_visualization_path_sub_folder = os.path.join(saved_visualization_path, current_subset_sign)
    
        os.makedirs(saved_visualization_path_sub_folder, exist_ok=True)


        if args.ablation_type == "local_finetuning_no_6_views":
            input_image = Image.open(input_image_path).convert('RGB')
            target_image = Image.open(target_image_path).convert('RGB')
            ref_image = Image.open(ref_image_path).convert('RGB')
            
            width,height = input_image.size
            
            with torch.no_grad():
                output_image = model.sample(
                    input_image,
                    height=288,
                    width=512,
                    ref_image=ref_image,
                    prompt=args.prompt
                )
            
            output_image = PIL_to_np(output_image)
            target_image = PIL_to_np(target_image)
            input_image = PIL_to_np(input_image)
            ref_image = PIL_to_np(ref_image)
            
            if args.vis:
                if total_folder_indicator%100==0:
                    stacked_images = stack_images_vertically(output_image, target_image, ref_image)
                    skimage.io.imsave(os.path.join(saved_visualization_path_sub_folder, f"{validation_key}.png"), stacked_images)


            psnr_left, ssim_left = compute_psnr_ssim(output_image, target_image)
            lpips_left = compute_lpips(output_image, target_image)
            all_metrics_dict[current_subset_sign]["psnr"] += psnr_left
            all_metrics_dict[current_subset_sign]["ssim"] += ssim_left
            all_metrics_dict[current_subset_sign]["lpips"] += lpips_left
            
            all_metrics_dict_counter[current_subset_sign] += 1
            
            all_metrics_dict["overall"]["psnr"] += psnr_left
            all_metrics_dict["overall"]["ssim"] += ssim_left
            all_metrics_dict["overall"]["lpips"] += lpips_left

            total_folder_indicator += 1

        
        elif args.ablation_type == "local_finetuning_all_training":
            input_image = Image.open(input_image_path).convert('RGB')
            target_image = Image.open(target_image_path).convert('RGB')
            ref_image = Image.open(ref_image_path).convert('RGB')
            
            width,height = input_image.size
            
            with torch.no_grad():
                output_image = model.sample(
                    input_image,
                    height=288,
                    width=512,
                    ref_image=ref_image,
                    prompt=args.prompt
                )
            
            output_image = PIL_to_np(output_image)
            target_image = PIL_to_np(target_image)
            input_image = PIL_to_np(input_image)
            ref_image = PIL_to_np(ref_image)
            
            if args.vis:
                if total_folder_indicator%100==0:
                    stacked_images = stack_images_vertically(output_image, target_image, ref_image)
                    skimage.io.imsave(os.path.join(saved_visualization_path_sub_folder, f"{validation_key}.png"), stacked_images)


            psnr_left, ssim_left = compute_psnr_ssim(output_image, target_image)
            lpips_left = compute_lpips(output_image, target_image)
            all_metrics_dict[current_subset_sign]["psnr"] += psnr_left
            all_metrics_dict[current_subset_sign]["ssim"] += ssim_left
            all_metrics_dict[current_subset_sign]["lpips"] += lpips_left
            
            all_metrics_dict_counter[current_subset_sign] += 1
            
            all_metrics_dict["overall"]["psnr"] += psnr_left
            all_metrics_dict["overall"]["ssim"] += ssim_left
            all_metrics_dict["overall"]["lpips"] += lpips_left

            total_folder_indicator += 1
            
        
        elif args.ablation_type == "local_finetuning_no_6_no_4_views":
            input_image = Image.open(input_image_path).convert('RGB')
            target_image = Image.open(target_image_path).convert('RGB')
            ref_image = Image.open(ref_image_path).convert('RGB')
            
            width,height = input_image.size
            
            with torch.no_grad():
                output_image = model.sample(
                    input_image,
                    height=288,
                    width=512,
                    ref_image=ref_image,
                    prompt=args.prompt
                )
            
            output_image = PIL_to_np(output_image)
            target_image = PIL_to_np(target_image)
            input_image = PIL_to_np(input_image)
            ref_image = PIL_to_np(ref_image)
            
            if args.vis:
                if total_folder_indicator%100==0:
                    stacked_images = stack_images_vertically(output_image, target_image, ref_image)
                    skimage.io.imsave(os.path.join(saved_visualization_path_sub_folder, f"{validation_key}.png"), stacked_images)


            psnr_left, ssim_left = compute_psnr_ssim(output_image, target_image)
            lpips_left = compute_lpips(output_image, target_image)
            all_metrics_dict[current_subset_sign]["psnr"] += psnr_left
            all_metrics_dict[current_subset_sign]["ssim"] += ssim_left
            all_metrics_dict[current_subset_sign]["lpips"] += lpips_left
            
            all_metrics_dict_counter[current_subset_sign] += 1
            
            all_metrics_dict["overall"]["psnr"] += psnr_left
            all_metrics_dict["overall"]["ssim"] += ssim_left
            all_metrics_dict["overall"]["lpips"] += lpips_left

            total_folder_indicator += 1
        
        
        elif args.ablation_type == "local_finetuning_all_training_no_ref":
            input_image = Image.open(input_image_path).convert('RGB')
            target_image = Image.open(target_image_path).convert('RGB')
            ref_image = Image.open(ref_image_path).convert('RGB')
            
            width,height = input_image.size
            
            with torch.no_grad():
                output_image = model.sample(
                    input_image,
                    height=288,
                    width=512,
                    ref_image=None,
                    prompt=args.prompt
                )
            
            output_image = PIL_to_np(output_image)
            target_image = PIL_to_np(target_image)
            input_image = PIL_to_np(input_image)
            ref_image = PIL_to_np(ref_image)
            
            if args.vis:
                if total_folder_indicator%100==0:
                    stacked_images = stack_images_vertically(output_image, target_image, ref_image)
                    skimage.io.imsave(os.path.join(saved_visualization_path_sub_folder, f"{validation_key}.png"), stacked_images)


            psnr_left, ssim_left = compute_psnr_ssim(output_image, target_image)
            lpips_left = compute_lpips(output_image, target_image)
            all_metrics_dict[current_subset_sign]["psnr"] += psnr_left
            all_metrics_dict[current_subset_sign]["ssim"] += ssim_left
            all_metrics_dict[current_subset_sign]["lpips"] += lpips_left
            
            all_metrics_dict_counter[current_subset_sign] += 1
            
            all_metrics_dict["overall"]["psnr"] += psnr_left
            all_metrics_dict["overall"]["ssim"] += ssim_left
            all_metrics_dict["overall"]["lpips"] += lpips_left

            total_folder_indicator += 1
        
        
        else:
            raise ValueError(f"Invalid ablation type: {args.ablation_type}")
        
        
    for key in all_metrics_dict:
        if key != "overall":
            all_metrics_dict[key]["psnr"] /= all_metrics_dict_counter[key]
            all_metrics_dict[key]["ssim"] /= all_metrics_dict_counter[key]
            all_metrics_dict[key]["lpips"] /= all_metrics_dict_counter[key]

        
    all_metrics_dict["overall"]["psnr"] /= total_folder_indicator
    all_metrics_dict["overall"]["ssim"] /= total_folder_indicator
    all_metrics_dict["overall"]["lpips"] /= total_folder_indicator

            
        
    
    save_dict_to_json(all_metrics_dict, args.output_dir + "all_metrics_dict.json")
        
    

    # # Process images
    # output_images = []
    # for i, input_image in enumerate(tqdm(input_images, desc="Processing images")):
    #     image = Image.open(input_image).convert('RGB')
    #     ref_image = Image.open(ref_images[i]).convert('RGB') if args.ref_image is not None else None
    #     output_image = model.sample(
    #         image,
    #         height=args.height,
    #         width=args.width,
    #         ref_image=ref_image,
    #         prompt=args.prompt
    #     )
    #     output_images.append(output_image)

    # # Save outputs
    # if args.video:
    #     # Save as video
    #     video_path = os.path.join(args.output_dir, "output.mp4")
    #     writer = imageio.get_writer(video_path, fps=30)
    #     for output_image in tqdm(output_images, desc="Saving video"):
    #         writer.append_data(np.array(output_image))
    #     writer.close()
    # else:
    #     # Save as individual images
    #     for i, output_image in enumerate(tqdm(output_images, desc="Saving images")):
    #         output_image.save(os.path.join(args.output_dir, os.path.basename(input_images[i])))