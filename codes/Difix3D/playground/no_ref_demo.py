import sys
sys.path.append("..")
import os
from diffusers.utils import load_image

from src.simplified_pipeline_diffix_no_ref import DifixPipeline



if __name__ == "__main__":
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")
    
    
    input_image_path = "/data4/zliu/Difix3D/KITTI360_Degradtations_Pairs/image/0/scene2013_05_28_drive_0000_sync_bin002_19.png"
    assert os.path.exists(input_image_path), "Input image path does not exist"
    
    input_image = load_image(input_image_path)
    prompt = "remove degradation"

    output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
    output_image.save("/home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/Difix3D/outputs/demo_output/from_pipeline_no_ref.png")