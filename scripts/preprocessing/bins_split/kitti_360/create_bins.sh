CREATE_BINS(){

cd ../../../..
cd preprocessing/bins_split/kitti360

root_path="/media/zliu/data12/dataset/KITTI/VSRD_Format/"
filelist_folder="/home/zliu/Desktop/Project2025/KITTI360_for_feedforward/FeedStereoGS/filenames/kitti360/avaliable_lists/"
min_bin_length=8.0
out_dir="/media/zliu/data12/dataset/KITTI/VSRD_Format/feedforward_bins/"


python create_bins.py \
            --root_path $root_path \
            --filelist_folder $filelist_folder \
            --min_bin_length $min_bin_length \
            --out_dir $out_dir



}


CREATE_BINS