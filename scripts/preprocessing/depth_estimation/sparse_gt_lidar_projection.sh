Project_GT_LiDAR_Into_Sparse_Depth(){
cd ../../..

root_folder="/data1/StereoDatasets/KITTI/KITTI360/"
output_folder="/data1/StereoDatasets/KITTI/KITTI360/projected_sparse_lidar/"

cd preprocessing/monocular_depth_estimation

CUDA_VISIBLE_DEVICES=2 python sparse_lidar_projection.py \
            --root_folder $root_folder \
            --output_folder $output_folder

}


Project_GT_LiDAR_Into_Sparse_Depth