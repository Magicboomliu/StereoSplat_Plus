CREATE_AVALIABLE_IMAGE_LIST(){

cd ../../../..
cd preprocessing/bins_split/kitti360

root_path="/media/zliu/data12/dataset/KITTI/VSRD_Format/"
output_path="/home/zliu/Desktop/Project2025/KITTI360_for_feedforward/FeedStereoGS/filenames/kitti360/avaliable_lists/"


python get_avaliable_images_list.py \
            --root_path $root_path \
            --output_path $output_path


}

CREATE_AVALIABLE_IMAGE_LIST