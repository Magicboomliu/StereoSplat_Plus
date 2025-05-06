python mono/tools/test_scale_cano.py \
    'mono/configs/HourglassDecoder/vit.raft5.giant2.py' \
    --load-from /data1/zliu/pretrained_foundataion_models/depth_estimation/Metric3Dv2/metric_depth_vit_giant2_800k.pth \
    --test_data_path /data2/zliu_backup_data/nuScenes/v1.0-mini/sweeps \
    --launcher None \
    --show-dir /data2/zliu_backup_data/nuScenes/v1.0-mini/sweeps_dptm