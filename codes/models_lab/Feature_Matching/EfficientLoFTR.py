from transformers import AutoImageProcessor, AutoModelForKeypointMatching
import torch
from PIL import Image
import requests




if __name__ == "__main__":
    
    # loading the image data
    image1 = "/data1/StereoDatasets/KITTI/KITTI360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000250.png"
    image2 = "/data1/StereoDatasets/KITTI/KITTI360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000260.png"
    image1 = Image.open(image1)
    image2 = Image.open(image2)

    # loading the depth data
    
    

    images = [image1, image2]
    processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
    model = AutoModelForKeypointMatching.from_pretrained("zju-community/efficientloftr")

    inputs = processor(images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to get keypoints and matches
    image_sizes = [[(image.height, image.width) for image in images]]
    processed_outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
    visualized_images = processor.visualize_keypoint_matching(images, processed_outputs)
    
    
    
    print(visualized_images[0].save("visualized_image1.png"))