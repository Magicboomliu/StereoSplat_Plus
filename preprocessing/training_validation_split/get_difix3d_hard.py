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


if __name__=="__main__":
    
    trainval_json_path = "/home/zliu/Project2025/OneStageTraining/FeedStereoGS/filenames/kitti360/difix_dataset/trainval.json"
    
    readed_traival_data = read_json_file(trainval_json_path)
    
    train_data = readed_traival_data['train']
    val_data   = readed_traival_data['test']
    
    # 阈值
    psnr_th  = 18.0
    ssim_th  = 0.6

    filtered_train_data = {}

    print("Start filtering train samples ...")
    for train_idx, data_contents in tqdm(train_data.items()):

        input_image_path  = data_contents['image']
        target_image_path = data_contents['target_image']
        ref_image_path    = data_contents['ref_image'] if 'ref_image' in data_contents else None
        
        # 这里用 input_image vs target_image 来算指标
        input_img  = read_image_file(input_image_path)
        target_img = read_image_file(target_image_path)

        cur_psnr = compute_psnr(input_img, target_img)
        cur_ssim = compute_ssim(input_img, target_img)

        # 保留 PSNR < 18 且 SSIM < 0.6 的样本
        if (cur_psnr < psnr_th) and (cur_ssim < ssim_th):
            # 你也可以把 PSNR/SSIM 存进去，方便之后分析
            data_contents['psnr'] = float(cur_psnr)
            data_contents['ssim'] = float(cur_ssim)
            filtered_train_data[train_idx] = data_contents

    print(f"Original train size: {len(train_data)}")
    print(f"Filtered  train size: {len(filtered_train_data)}")

    # 重新组织 json，test 原样保留
    filtered_data = {
        "train": filtered_train_data,
        "test":  val_data
    }

    # 输出路径
    out_json_path = os.path.join(
        os.path.dirname(trainval_json_path),
        "trainval_hard.json"
    )

    with open(out_json_path, "w") as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Filtered json saved to: {out_json_path}")