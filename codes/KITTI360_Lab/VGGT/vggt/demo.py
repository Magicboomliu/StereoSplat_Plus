import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
path = "/home/zliu/Project2025/examples/"
# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
image_names =sorted([os.path.join(path,p) for p in os.listdir(path)]) 
images = load_and_preprocess_images(image_names).to(device)


with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        depth = predictions['depth']
        pose_enc = predictions["pose_enc"]
        images = predictions['images']
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
 

        print(extrinsic[0][0][:3,3])
        print(extrinsic[0][1][:3,3])
        print(extrinsic[0][2][:3,3])
        print(extrinsic[0][3][:3,3])
        print(extrinsic[0][4][:3,3])
        print(extrinsic[0][5][:3,3])
        print(extrinsic[0][6][:3,3])
        print(extrinsic[0][7][:3,3])

        quit()
        
