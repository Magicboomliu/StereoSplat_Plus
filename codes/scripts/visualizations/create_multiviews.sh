Get_Output_Images(){

cd ../..
accelerate launch get_supervied_images_depths_aplha.py \
    --py-config configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py \
    --output-dir /home/zliu/Project2025/Omni-Scene/Outputs_Results/output_predicted_views/omni_gs_nusc_novelview_r50_224x400_vis \
    --load-from /data2/zliu_backup_data/FeedForwardGS_Datasest/nuscence/checkpoint-100000

}

Get_Output_Images