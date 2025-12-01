
RUN_THE_DIFIX_INFRENCE(){
cd ..


model_name="nvidia/difix_ref"
model_path=None
input_image_path="/data1/zliu/KITTI360_Completed/KITTI360_Degradtations_Pairs/image/0/scene2013_05_28_drive_0000_sync_bin002_22.png"
ref_image_path="/data1/zliu/KITTI360_Completed/KITTI360_Degradtations_Pairs/ref_image/0/scene2013_05_28_drive_0000_sync_bin002_22.png"
prompt="remove degradation"
output_dir="outputs/demo_output"
timestep=399
guidance_scale=5.0


python src/inference_difix.py \
    --model_name $model_name \
    --model_path $model_path \
    --input_image $input_image_path \
    --ref_image $ref_image_path \
    --prompt "remove degradation" \
    --output_dir $output_dir \
    --timestep $timestep \

}

RUN_THE_DIFIX_INFRENCE