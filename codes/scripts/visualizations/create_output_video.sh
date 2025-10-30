
Create_Videos(){

cd ../..

accelerate launch  demo.py \
    --py-config configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py \
    --output-dir /home/zliu/Project2025/Omni-Scene/Outputs_Results/rendered_videos/omni_gs_nusc_novelview_r50_224x400_vis \
    --load-from /data2/zliu_backup_data/FeedForwardGS_Datasest/nuscence/checkpoint-100000
}

Create_Videos

