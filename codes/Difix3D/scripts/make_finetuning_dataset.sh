Make_Near_View_Finetuning_Dataset(){

cd ..
cd playground

root_folder="/data1/zliu/IROS26/Difix3D_Pose_Prompt/"
dataset_type='Validation_Set' # 'training' or 'validation'
output_filename_folder='/home/zliu/IROS2026/StereoSplat_Plus/codes/Difix3D/filenames/'
psnr_threshold=20.0


python make_near_view_finetuning.py \
    --root_folder $root_folder \
    --dataset_type $dataset_type \
    --output_filename_folder $output_filename_folder \
    --psnr_threshold $psnr_threshold



}


Make_Near_View_Finetuning_Dataset