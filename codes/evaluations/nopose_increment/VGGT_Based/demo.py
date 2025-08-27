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
images = load_and_preprocess_images(image_names).to(device) #(V,3,H，W)]


with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        depth = predictions['depth'] 
        depth_conf = predictions['depth_conf']
        pose_enc = predictions["pose_enc"]
        images = predictions['images']
        pcd = predictions['world_points']
        pcd_conf = predictions['world_points_conf']
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        
        print(depth.shape) # torch.Size([1, 16, 140, 518, 1])
        print(depth_conf.shape) # torch.Size([1, 16, 140, 518, 1])
        print(pose_enc.shape) # torch.Size([1, 16, 140, 518, 1])
        print(images.shape) # torch.Size([1, 3, 140, 518])
        print(pcd.shape) # torch.Size([1, 16, 140, 518, 3])
        print(pcd_conf.shape) # torch.Size([1, 16, 140, 518, 1])
        print(extrinsic.shape) # torch.Size([1, 16, 4, 4])
        print(intrinsic.shape) # torch.Size([1, 16, 3, 3])
        quit()


        
