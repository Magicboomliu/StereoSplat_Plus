import os
import cv2

def resize_images_half(input_folder, output_folder):
    # 支持的图片扩展名
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_exts:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取图片
            img = cv2.imread(input_path)
            if img is None:
                print(f"Failed to read: {input_path}")
                continue

            # Resize 为原图一半大小
            h, w = img.shape[:2]
            resized = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

            # 保存结果
            cv2.imwrite(output_path, resized)
            print(f"Saved: {output_path}")

# 使用方式示例：
resize_images_half("/home/zliu/Project2025/examples/images_full", "/home/zliu/Project2025/examples/images")
