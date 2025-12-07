RUN_Difix_Model_Evaluation_ON_KITTI360_Testset(){
cd ..

input_test_filename="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/filenames/kitti360/difix_dataset/trainval.json"
model_name="nvidia/difix_ref"
model_path="/data4/zliu/Difix3D/Pretrained_Models/Fintune_Difix_Ref_all/checkpoints/model_210001.pkl"
ablation_study_name="finetune_difix_ref_all"
seed=42
timestep=199
use_model_type="local"
output_folder="/data4/zliu/Difix3D/Evaluations20251207/$ablation_study_name"
height=112
width=544

python src/diffix_model_evaluation.py \
    --input_test_filename $input_test_filename \
    --model_name $model_name \
    --model_path $model_path \
    --ablation_study_name $ablation_study_name \
    --output_folder $output_folder \
    --seed $seed \
    --timestep $timestep \
    --use_model_type $use_model_type \
    --use_ref \
    --height $height \
    --width $width \
    # --vis

}


RUN_Difix_Model_Evaluation_ON_KITTI360_Testset