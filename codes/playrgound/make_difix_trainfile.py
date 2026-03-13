import os
from pathlib import Path
from typing import List
from tqdm import tqdm


import torch
import numpy as np
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure


def load_image_as_tensor(image_path):
    """
    读取图片并转成 torch tensor: [1, 3, H, W], range [0, 1]
    """
    img = Image.open(image_path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return img


def compute_psnr(est, gt, eps=1e-8):
    """
    est, gt: [1, 3, H, W], range [0, 1]
    """
    mse = torch.mean((est - gt) ** 2)
    psnr = -10.0 * torch.log10(mse + eps)
    return psnr.item()


def compute_ssim(est, gt):
    """
    est, gt: [1, 3, H, W], range [0, 1]
    """
    ssim = structural_similarity_index_measure(est, gt, data_range=1.0)
    return ssim.item()






def collect_image_paths(root_dir: str | Path,
                        exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")) -> List[str]:
    """
    递归遍历 root_dir（包含子文件夹），收集所有图片的绝对路径。
    """
    root = Path(root_dir)
    image_paths: List[str] = []

    if not root.exists():
        raise FileNotFoundError(f"Root dir not found: {root}")

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(exts):
                abs_path = Path(dirpath) / name
                image_paths.append(str(abs_path.resolve()))

    return image_paths


def main() -> None:
    threshold = 15
    cnt = 0
    # 用户给定的绝对路径（如果路径有误，可在这里修改）
    rendered_root_dir = "/data1/zliu/IROS26/Compared_With_Others/Diff-StereoSplat/difix_finetuning_dataset/rendered_views/"
    gt_root_dir = "/data1/zliu/IROS26/Compared_With_Others/Diff-StereoSplat/difix_finetuning_dataset/gt_views/"
    
    rendered_images_list = collect_image_paths(rendered_root_dir)
    gt_images_list = []
    
    # 使用 set 保证场景名唯一，不重复
    bad_scenes = set()
    all_scenes = set()
    
    # remove all the bad scences
    for fname in tqdm(rendered_images_list):
        gt_name = fname.replace("rendered","gt")
        assert os.path.exists(gt_name)
        gt_images_list.append(gt_name)
        
        
        img1 = load_image_as_tensor(fname)
        img2 = load_image_as_tensor(gt_name)

        # 保证尺寸一致
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes do not match: {img1.shape} vs {img2.shape}")

        psnr = compute_psnr(img1, img2)
        ssim = compute_ssim(img1, img2)
        
        scene_name = fname.split("/")[-2]

        
        if psnr < threshold:
            cnt += 1
            bad_scenes.add(scene_name)
            
        all_scenes.add(scene_name)
    

    with open("difix_trainfile_all.txt", "w") as f:
        for idx in range(len(rendered_images_list)):
            rendered_fname = rendered_images_list[idx]
            gt_fname = gt_images_list[idx]
            if idx != len(rendered_images_list)-1:
                f.write(rendered_fname + " " + gt_fname + "\n")
            else:
                f.write(rendered_fname + " " + gt_fname)
    
    
    new_rendered_images_list = []
    new_gt_images_list = []
    
    print(len(rendered_images_list))
    print(len(gt_images_list))
    print("--------------------------------")
    
    for idx in range(len(rendered_images_list)):
        rendered_fname = rendered_images_list[idx]
        gt_fname = gt_images_list[idx]
        
        scene_name_rendered = rendered_fname.split("/")[-2]
        scene_name_gt = gt_fname.split("/")[-2]
        
        assert scene_name_rendered == scene_name_gt
        
        if scene_name_rendered not in bad_scenes:
            new_rendered_images_list.append(rendered_fname)
            new_gt_images_list.append(gt_fname)
        

    
    print(len(new_rendered_images_list))
    print(len(new_gt_images_list))
    print("--------------------------------")

    
    with open("difix_trainfile_safe_{}.txt".format(threshold), "w") as f:
        for idx in range(len(new_rendered_images_list)):
            rendered_fname = new_rendered_images_list[idx]
            gt_fname = new_gt_images_list[idx]
            if idx != len(new_rendered_images_list)-1:
                f.write(rendered_fname + " " + gt_fname + "\n")
            else:
                f.write(rendered_fname + " " + gt_fname)

    
    
    

    






if __name__ == "__main__":
    main()