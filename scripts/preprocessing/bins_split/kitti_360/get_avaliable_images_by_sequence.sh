CREATE_AVALIABLE_IMAGE_LIST(){

cd ../../../..
cd preprocessing/bins_split/kitti360

root_path="/data1/StereoDatasets/KITTI/KITTI360/"
output_path="/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/avaliable_lists/"


python get_avaliable_images_list.py \
            --root_path $root_path \
            --output_path $output_path


}

CREATE_AVALIABLE_IMAGE_LIST