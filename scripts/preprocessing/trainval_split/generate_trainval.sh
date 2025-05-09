Generate_TrainVal_Splits(){
cd ../../..

cd preprocessing/bins_split/kitti360/trainval_split/

feedforward_bin_path="/data1/StereoDatasets/KITTI/KITTI360/feedforward_bins/bin_infos_8.0/"
sequence_name="2013_05_28_drive_0000_sync" # "2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync","2013_05_28_drive_0004_sync"
# "2013_05_28_drive_0005_sync" "2013_05_28_drive_0006_sync" "2013_05_28_drive_0007_sync" "2013_05_28_drive_0009_sync" "2013_05_28_drive_0010_sync0"
# "all"
output_folder="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval"
seed=1024
val_ratio=0.1

python split_trainval.py \
        --feedforward_bin_path $feedforward_bin_path \
        --sequence_name $sequence_name \
        --output_folder $output_folder \
        --seed $seed \
        --val_ratio $val_ratio
 
}


Generate_TrainVal_Splits