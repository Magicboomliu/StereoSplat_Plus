import json
import torch
from PIL import Image
import torchvision.transforms.functional as F
import os
import numpy as np


class KITTI360_Restoration_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, 
                 split, 
                 height=576, 
                 width=1024, 
                 tokenizer=None,
                 use_relative_pose=False):
        super().__init__()
        with open(dataset_path, "r") as f:
            payload = json.load(f)
        
        key = split
        if key == "train" and "train" not in payload and "training" in payload:
            key = "training"
        elif key == "test" and "test" not in payload and "validation" in payload:
            key = "validation"
        self.data = payload[key]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer
        
        self.use_relative_pose = use_relative_pose
        
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        
        input_img = self.data[img_id]["image"]
        output_img = self.data[img_id]["target_image"]
        ref_img = self.data[img_id]["ref_image"] if "ref_image" in self.data[img_id] else None
        caption = self.data[img_id]["prompt"]
        
        try:
            input_img = Image.open(input_img)
            output_img = Image.open(output_img)
        except:
            print("Error loading image:", input_img, output_img)
            return self.__getitem__(idx + 1)

        img_t = F.to_tensor(input_img)
        img_t = F.resize(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        output_t = F.to_tensor(output_img)
        output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        if ref_img is not None:
            
            ref_img = Image.open(ref_img)
            ref_t = F.to_tensor(ref_img)
            ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
            img_t = torch.stack([img_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0) 
                       
        else:
            img_t = img_t.unsqueeze(0)
            output_t = output_t.unsqueeze(0)

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
        }
        
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids
        
        
        if self.use_relative_pose:
            relative_pose = self.data[img_id]["relative_pose"]
        
            if ref_img is not None:            
                relative_pose_first_pose = np.eye(4, dtype=np.float32)
                relative_pose = np.stack([relative_pose, relative_pose_first_pose], axis=0)
    
            out["relative_pose"] = torch.from_numpy(np.array(relative_pose))
    
        return out






# Paired Dataset with raw images/target/ref images and relative pose
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None):

        super().__init__()
        with open(dataset_path, "r") as f:
            payload = json.load(f)
        # 与 make_near_view_finetuning 等脚本对齐：training/validation ↔ train/test
        key = split
        if key == "train" and "train" not in payload and "training" in payload:
            key = "training"
        elif key == "test" and "test" not in payload and "validation" in payload:
            key = "validation"
        self.data = payload[key]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer
        

    def __len__(self):

        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        
        input_img = self.data[img_id]["image"]
        output_img = self.data[img_id]["target_image"]
        ref_img = self.data[img_id]["ref_image"] if "ref_image" in self.data[img_id] else None
        caption = self.data[img_id]["prompt"]
        
        try:
            input_img = Image.open(input_img)
            output_img = Image.open(output_img)
        except:
            print("Error loading image:", input_img, output_img)
            return self.__getitem__(idx + 1)

        img_t = F.to_tensor(input_img)
        img_t = F.resize(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        output_t = F.to_tensor(output_img)
        output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        if ref_img is not None:
            ref_img = Image.open(ref_img)
            ref_t = F.to_tensor(ref_img)
            ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
        
            img_t = torch.stack([img_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0)            
        else:
            img_t = img_t.unsqueeze(0)
            output_t = output_t.unsqueeze(0)

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
        }
        
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out




if __name__ == "__main__":
    
    use_relative_pose = True
    
    dataset_path = "/home/zliu/IROS2026/StereoSplat_Plus/codes/Difix3D/filenames/Validation_Set/all_results_dict.json"
    assert os.path.exists(dataset_path)
    
    kitti360_restoration_dataset = KITTI360_Restoration_Dataset(dataset_path, 
                                                                "train", 
                                                                height=112, 
                                                                width=544, 
                                                                tokenizer=None,
                                                                use_relative_pose=use_relative_pose)
    
    
    
    for data in kitti360_restoration_dataset:
        # dict_keys(['output_pixel_values', 'conditioning_pixel_values', 'caption'])
    
        raw_images_with_ref = data['conditioning_pixel_values']
        target_image = data['output_pixel_values']
        caption = data['caption']
        if "relative_pose" in data.keys():        
            relative_pose = data['relative_pose']
            print(relative_pose.shape)
        
        print(raw_images_with_ref.shape)
        print(target_image.shape)
        print(caption)
        
        
        quit()
        
