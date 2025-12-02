import json
import os
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def read_json_file(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)
    
    
def read_image_file(image_file_path):
    # 统一转成 RGB 并归一化到 [0, 1]
    img = Image.open(image_file_path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    return img


def compute_ssim(image1, image2):
    # skimage 的 ssim 没有 size_average 参数，用 channel_axis 指定最后一维为通道
    return ssim(image1, image2, data_range=1.0, channel_axis=-1)


def compute_psnr(image1, image2):
    return psnr(image1, image2, data_range=1.0)


if __name__ == "__main__":
    
    totally_trainval_contents_path = "/home/zliu/Project2025/OneStageTraining/FeedStereoGS/filenames/kitti360/difix_dataset/trainval.json"
    hard_trainval_contents_path = "/home/zliu/Project2025/OneStageTraining/FeedStereoGS/filenames/kitti360/difix_dataset/trainval_hard.json"
    
    totally_trainval_contents = read_json_file(totally_trainval_contents_path)
    hard_trainval_contents = read_json_file(hard_trainval_contents_path)
    
    
    total_train_contents = totally_trainval_contents['train']
    hard_train_contents = hard_trainval_contents['train']
    
    
    val_data = totally_trainval_contents['test']
    

    filtered_train_data = {}
    
    for key in tqdm(total_train_contents.keys()):
        if key not in hard_train_contents.keys():
            filtered_train_data[key] = total_train_contents[key]
        

    print(f"Original train size: {len(total_train_contents)}")
    print(f"Filtered  train size: {len(filtered_train_data)}")

    # 重新组织 json，test 原样保留
    filtered_data = {
        "train": filtered_train_data,
        "test":  val_data
    }

    # 输出路径
    out_json_path = os.path.join(
        os.path.dirname(totally_trainval_contents_path),
        "trainval_easy_plus_medium.json"
    )

    with open(out_json_path, "w") as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Filtered json saved to: {out_json_path}")
    
