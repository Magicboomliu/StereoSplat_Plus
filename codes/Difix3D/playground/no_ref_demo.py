import sys
sys.path.append("..")


from src.pipeline_difix import DifixPipeline
from diffusers.utils import load_image


if __name__ == "__main__":
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")

    input_image = load_image("/home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/Difix3D/assets/example_input.png")
    prompt = "remove degradation"

    output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
    output_image.save("/home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/Difix3D/assets/example_output.png")