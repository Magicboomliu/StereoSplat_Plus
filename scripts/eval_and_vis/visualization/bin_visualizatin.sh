Bin_Data_Visualization(){

cd ../../..
cd preprocessing/bins_split/kitti360
dataroot="/media/zliu/data12/dataset/KITTI/VSRD_Format/"
version="bin_infos_8.0"
# bin_list="/home/zliu/Desktop/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt"
bin_list="None"
single_bin_token="/media/zliu/data12/dataset/KITTI/VSRD_Format/feedforward_bins/bin_infos_8.0/scene2013_05_28_drive_0000_sync_bin102.pkl"
output_folder="/home/zliu/Desktop/Project2025/FeedStereoGS/temp/feedstereo_outputs/omni_gs_kitti360_novelview_r50_224x840/inputs_bins"
# add_pseudo_depth



python bin_data_visualization.py --dataroot $dataroot \
                                 --version $version \
                                 --bin_list $bin_list \
                                 --single_bin_token $single_bin_token \
                                 --output_folder $output_folder \
                                 --add_pseudo_depth


}

Bin_Data_Visualization