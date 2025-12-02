import sys
sys.path.append("..")
from diffusers.utils import load_image

import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src.simplified_pipeline_difix import DifixPipeline


def compute_psnr_and_ssim(image1, image2):
    """
    Compute PSNR and SSIM between two images.
    
    Args:
        image1: First image (PIL Image or numpy array)
        image2: Second image (PIL Image or numpy array)
    
    Returns:
        tuple: (psnr, ssim) values
    """
    # Convert PIL images to numpy arrays if needed
    if hasattr(image1, 'mode'):
        image1 = np.array(image1)
    if hasattr(image2, 'mode'):
        image2 = np.array(image2)
    
    # Ensure images are uint8 and in range [0, 255]
    if image1.dtype != np.uint8:
        image1 = (image1 * 255).astype(np.uint8) if image1.max() <= 1.0 else image1.astype(np.uint8)
    if image2.dtype != np.uint8:
        image2 = (image2 * 255).astype(np.uint8) if image2.max() <= 1.0 else image2.astype(np.uint8)
    
    # Compute PSNR
    psnr = peak_signal_noise_ratio(image1, image2, data_range=255)
    
    # Compute SSIM (for skimage >=0.19 use channel_axis=-1, older versions: multichannel=True)
    ssim = structural_similarity(image1, image2, data_range=255, channel_axis=-1)
    
    return psnr, ssim



if __name__ == "__main__":

    pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    pipe.to("cuda")
        
    
    input_image_path = "/data1/zliu/KITTI360_Completed/KITTI360_Degradtations_Pairs/image/0/scene2013_05_28_drive_0000_sync_bin002_22.png"
    ref_image_path = "/data1/zliu/KITTI360_Completed/KITTI360_Degradtations_Pairs/ref_image/0/scene2013_05_28_drive_0000_sync_bin002_22.png"
    target_image_path = "/data1/zliu/KITTI360_Completed/KITTI360_Degradtations_Pairs/target_image/0/scene2013_05_28_drive_0000_sync_bin002_22.png"

    assert os.path.exists(input_image_path), "Input image path does not exist"
    assert os.path.exists(ref_image_path), "Ref image path does not exist"
    assert os.path.exists(target_image_path), "Target image path does not exist"
    
    
    input_image = load_image(input_image_path)
    ref_image = load_image(ref_image_path)
    target_image = load_image(target_image_path)
    
    remove_degradation_prompt = "remove degradation"

    # Timesteps to explore
    timesteps_list = [199, 299, 399, 499, 599]
    
    # Store results
    results = {
        'timesteps': [],
        'psnr': [],
        'ssim': [],
        'images': []
    }
    
    # Compute baseline metrics
    input_image_psnr, input_image_ssim = compute_psnr_and_ssim(input_image, target_image)
    ref_image_psnr, ref_image_ssim = compute_psnr_and_ssim(ref_image, target_image)
    

    timesteps = 499
    guidance_scale = 0.0

    enhanced_image = pipe(remove_degradation_prompt, image=input_image, 
                        ref_image=ref_image, num_inference_steps=10, 
                        timesteps=[timesteps], 
                        guidance_scale=guidance_scale)
    
    
    saved_path_from_difix_pipeline = "/home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/Difix3D/outputs/demo_output/from_pipeline.png"
    enhanced_image.save(saved_path_from_difix_pipeline)
    
    
    
    enhanced_image_psnr, enhanced_image_ssim = compute_psnr_and_ssim(enhanced_image, target_image)
    input_image_psnr, input_image_ssim = compute_psnr_and_ssim(input_image, target_image)
    
    print(f"Enhanced Image PSNR: {enhanced_image_psnr:.4f}, SSIM: {enhanced_image_ssim:.4f}")
    print(f"Input Image PSNR: {input_image_psnr:.4f}, SSIM: {input_image_ssim:.4f}")
    
    
