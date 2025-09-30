import os
import argparse
import numpy as np

from pipeline_difix import DifixPipeline
from diffusers.utils import load_image





if __name__ == "__main__":

    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")

    input_image = load_image("/home/zliu/Desktop/Project2025/Difix3D/FeedStereoGS/assets/center_stereo.png")
    
    prompt = "remove degradation"
    ref_image = load_image("/home/zliu/Desktop/Project2025/Difix3D/FeedStereoGS/assets/first_stereo_ref.png")
    
    ref_image = ref_image.resize((input_image.size[0], input_image.size[1]))


    output_image = pipe(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
    output_image.save("/home/zliu/Desktop/Project2025/Difix3D/FeedStereoGS/assets/ref_based_kitti_enhanced.png")