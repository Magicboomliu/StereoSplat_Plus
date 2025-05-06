GET_RELATIVE_DEPTH(){

inputdir="/data2/zliu_backup_data/nuScenes/v1.0-mini/sweeps/"
outdir="/data2/zliu_backup_data/nuScenes/v1.0-mini/sweeps_dpt/"
encoder='vitl'
checkpoint_path="/data1/zliu/pretrained_foundataion_models/depth_estimation/DepthAnythingV2/depth_anything_v2_vitl.pth"


python get_relative_depth_results_seq.py --inputdir $inputdir \
                                        --outdir $outdir \
                                        --encoder $encoder \
                                        --checkpoint_path $checkpoint_path


}

GET_RELATIVE_DEPTH