import os
from PIL import Image

def images_to_gif(folder, output_path="output.gif", duration=250):
    """
    将文件夹中的所有图片按文件名顺序拼接成 GIF 动画

    Args:
        folder: 图片文件夹路径
        output_path: 输出 gif 文件路径
        duration: 每帧显示时间（毫秒），例如 100ms = 10 FPS
    """
    # 读取图片文件列表并排序
    image_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        raise ValueError("No images found in folder.")

    # 加载所有图片
    images = [Image.open(os.path.join(folder, f)) for f in image_files]

    # 保存为 GIF，loop=0 表示循环播放
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"✅ GIF saved to {output_path}")
    

if __name__=="__main__":
    types = "est_depths/monodepthv2"
    
    input_folder = "/home/zliu/Desktop/Project2025/FeedStereoGS/temp/feedstereo_outputs/omni_gs_kitti360_novelview_r50_224x840/inputs_bins/scene2013_05_28_drive_0000_sync_bin102/"
    output_path = "/home/zliu/Desktop/Project2025/FeedStereoGS/temp/feedstereo_outputs/omni_gs_kitti360_novelview_r50_224x840/inputs_bins/scene2013_05_28_drive_0000_sync_bin102/GIFS"
    os.makedirs(output_path,exist_ok=True)
    
    images_to_gif(os.path.join(input_folder,types),os.path.join(output_path,types)+".gif")
    
    pass