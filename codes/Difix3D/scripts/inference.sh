
RUN_THE_DIFIX_WITH_REF_INFRENCE(){
cd ..


model_name="nvidia/difix_ref"
model_path="/data4/zliu/Difix3D/Pretrained_Models/Fintune_Difix_Ref_All/checkpoints/model_1.pkl"
input_image_path="/data4/zliu/Difix3D/KITTI360_Degradtations_Pairs/image/0/scene2013_05_28_drive_0000_sync_bin002_19.png"
ref_image_path="/data4/zliu/Difix3D/KITTI360_Degradtations_Pairs/ref_image/0/scene2013_05_28_drive_0000_sync_bin002_19.png"
prompt="remove degradation"
output_dir="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/Difix3D/outputs/demo_output"
timestep=399
guidance_scale=5.0
ablation_name="official_difix_ref"
use_weight_format="ckpt"


python src/inference_difix_with_ref.py \
    --model_name $model_name \
    --model_path $model_path \
    --input_image $input_image_path \
    --ref_image $ref_image_path \
    --prompt "remove degradation" \
    --output_dir $output_dir \
    --timestep $timestep \
    --ablation_name $ablation_name \
    --use_weight_format $use_weight_format
}



RUN_THE_DIFIX_WITHOUT_REF_INFRENCE(){
cd ..


model_name="nvidia/difix"
model_path=None
input_image_path="/data4/zliu/Difix3D/KITTI360_Degradtations_Pairs/image/0/scene2013_05_28_drive_0000_sync_bin002_19.png"
ref_image_path="/data4/zliu/Difix3D/KITTI360_Degradtations_Pairs/ref_image/0/scene2013_05_28_drive_0000_sync_bin002_19.png"
prompt="remove degradation"
output_dir="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/Difix3D/outputs/demo_output"
timestep=399
guidance_scale=5.0
ablation_name="official_difix"
use_weight_format="huggingface"

python src/inference_difix_no_ref.py \
    --model_name $model_name \
    --model_path $model_path \
    --input_image $input_image_path \
    --prompt "remove degradation" \
    --output_dir $output_dir \
    --timestep $timestep \
    --ablation_name $ablation_name \
    --use_weight_format $use_weight_format

}


RUN_THE_DIFIX_WITH_REF_INFRENCE
# RUN_THE_DIFIX_WITHOUT_REF_INFRENCE
