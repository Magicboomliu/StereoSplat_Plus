import sys
sys.path.append("..")
from src.pipeline_difix import DifixPipeline
from diffusers.utils import load_image

import os
from matplotlib import pyplot as plt


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


    enhanced_image = pipe(remove_degradation_prompt, image=input_image, 
                        ref_image=ref_image, num_inference_steps=1, 
                        timesteps=[199], 
                        guidance_scale=0.0).images[0]
    

    
    plt.subplot(4, 1, 1)
    plt.axis("off")
    plt.title("Input Image")
    plt.imshow(input_image)
    plt.subplot(4, 1, 2)
    plt.axis("off")
    plt.title("Ref Image")
    plt.imshow(ref_image)
    plt.subplot(4, 1, 3)
    plt.axis("off")
    plt.title("Enhanced Image")
    plt.imshow(enhanced_image)
    plt.subplot(4, 1, 4)
    plt.axis("off")
    plt.title("Target Image")
    plt.imshow(target_image)
    plt.savefig("all_to_images.png")
    
    # output_image.save("enhanced_image.png")
    
