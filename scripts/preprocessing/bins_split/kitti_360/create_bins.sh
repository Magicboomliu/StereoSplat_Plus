CREATE_BINS(){

cd ../../../..
cd preprocessing/bins_split/kitti360

root_path="/data1/StereoDatasets/KITTI/KITTI360/"
filelist_folder="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/avaliable_lists/"
min_bin_length=8.0
out_dir="/data1/StereoDatasets/KITTI/KITTI360/feedforward_bins/"


python create_bins.py \
            --root_path $root_path \
            --filelist_folder $filelist_folder \
            --min_bin_length $min_bin_length \
            --out_dir $out_dir



}


CREATE_BINS