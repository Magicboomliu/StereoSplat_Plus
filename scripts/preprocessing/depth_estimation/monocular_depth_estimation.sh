Monocular_Depth_Estimation_With_DepthAnythingV2()
{
cd ../../..

cd preprocessing/monocular_depth_estimation/Depth-Anything-V2


sequence_name="2013_05_28_drive_0000_sync" # All mean\s all
root_path="/data1/StereoDatasets/KITTI/KITTI360/"
out_path="/data1/StereoDatasets/KITTI/KITTI360/monocular_depth/monodepthV2/"
encoder='vitl'
checkpoint_path="/data1/zliu/pretrained_foundataion_models/depth_estimation/DepthAnythingV2/depth_anything_v2_vitl.pth"


CUDA_VISIBLE_DEVICES=1 python get_rel_depth.py --root_path $root_path \
                                        --out_path $out_path \
                                        --encoder $encoder \
                                        --checkpoint_path $checkpoint_path \
                                        --sequence_name $sequence_name

}



# Monodepth estimation using MonoDepthV2
Monocular_Depth_Estimation_With_DepthAnythingV2