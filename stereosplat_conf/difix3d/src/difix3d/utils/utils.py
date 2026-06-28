import numpy as np
import torch
import torch.nn.functional as F

def Convert_Tensor_to_Image(tensor):
    tensor = torch.clamp(tensor, min=-1.0, max=1.0)
    image_tensor = tensor.detach().squeeze(0).squeeze(0).permute(1,2,0).cpu().numpy() 
    image_tensor = image_tensor * 0.5 + 0.5
    image_tensor = (image_tensor * 255).astype(np.uint8)
    return image_tensor