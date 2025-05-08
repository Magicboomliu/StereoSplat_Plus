
# create the annotations file for depth infernece
STEP1(){

root_path="/data1/StereoDatasets/KITTI/KITTI360/"
out_json_folder="/home/zliu/Project2025/FeedStereoGS/preprocessing/monocular_depth_estimation/Metric3D/data/kitti360_demo"
sequence_name="2013_05_28_drive_0000_sync"
CUDA_VISIBLE_DEVICES=2 python mono/tools/kitti360_dataset_format_organization.py \
            --root_path $root_path \
            --out_path $out_json_folder \
            --sequence_name $sequence_name

}

# conduct the confidence and the depth estimation
STEP2(){
CUDA_VISIBLE_DEVICES=2 python mono/tools/test_scale_kitti360.py \
    '/home/zliu/Project2025/FeedStereoGS/preprocessing/monocular_depth_estimation/Metric3D/mono/configs/HourglassDecoder/test_kitti_convlarge.0.3_150.py' \
    --load-from /data1/zliu/pretrained_foundataion_models/depth_estimation/Metric3Dv2/convlarge_hourglass_0.3_150_step750k_v1.1.pth \
    --test_data_path /home/zliu/Project2025/FeedStereoGS/preprocessing/monocular_depth_estimation/Metric3D/data/kitti360_demo/test_annotations_2013_05_28_drive_0000_sync.json \
    --show_dir "/data1/StereoDatasets/KITTI/KITTI360/monocular_depth/Metric3DV2/" \
    --launcher None \
    --dataset_root_path "/data1/StereoDatasets/KITTI/KITTI360/"
}

# STEP1
STEP2