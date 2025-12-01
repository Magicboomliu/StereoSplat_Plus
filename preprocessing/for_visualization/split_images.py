import os
import shutil
import numpy as np
import cv2
import skimage.io
import glob




if __name__ == "__main__":
    
    GT_Images_Folder = "/data1/zliu/forward_outputs_compared_with_others/mvsplat/forward_views/scene2013_05_28_drive_0000_sync_bin008/GT_Images/"
    
    saved_splited_images_folder = "splited_images"
    os.makedirs(saved_splited_images_folder, exist_ok=True)
    
    
    first_concated_stereo_image_path = os.path.join(GT_Images_Folder, "first_stereo.png")
    last_concated_stereo_image_path = os.path.join(GT_Images_Folder, "last_stereo.png")
    center_concated_stereo_image_path = os.path.join(GT_Images_Folder, "center_stereo.png")
    
    first_concated_stereo_image = skimage.io.imread(first_concated_stereo_image_path)
    last_concated_stereo_image = skimage.io.imread(last_concated_stereo_image_path)
    center_concated_stereo_image = skimage.io.imread(center_concated_stereo_image_path)
    
    
    full_width = first_concated_stereo_image.shape[1]
    
    width_half = full_width // 2
    

    
    frist_stereo_left = first_concated_stereo_image[:, :width_half, :]
    frist_stereo_right = first_concated_stereo_image[:, width_half:, :]
    last_stereo_left = last_concated_stereo_image[:, :width_half, :]
    last_stereo_right = last_concated_stereo_image[:, width_half:, :]
    center_stereo_left = center_concated_stereo_image[:, :width_half, :]
    center_stereo_right = center_concated_stereo_image[:, width_half:, :]
    
    
    saved_first_stereo_left_image_path = os.path.join(saved_splited_images_folder, "first_stereo_left.png")
    saved_first_stereo_right_image_path = os.path.join(saved_splited_images_folder, "first_stereo_right.png")
    saved_last_stereo_left_image_path = os.path.join(saved_splited_images_folder, "last_stereo_left.png")
    saved_last_stereo_right_image_path = os.path.join(saved_splited_images_folder, "last_stereo_right.png")
    saved_center_stereo_left_image_path = os.path.join(saved_splited_images_folder, "center_stereo_left.png")
    saved_center_stereo_right_image_path = os.path.join(saved_splited_images_folder, "center_stereo_right.png")
    
    skimage.io.imsave(saved_first_stereo_left_image_path, frist_stereo_left)
    skimage.io.imsave(saved_first_stereo_right_image_path, frist_stereo_right)
    skimage.io.imsave(saved_last_stereo_left_image_path, last_stereo_left)
    skimage.io.imsave(saved_last_stereo_right_image_path, last_stereo_right)
    skimage.io.imsave(saved_center_stereo_left_image_path, center_stereo_left)
    skimage.io.imsave(saved_center_stereo_right_image_path, center_stereo_right)
    
