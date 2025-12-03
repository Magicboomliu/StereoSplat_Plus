# One Step Diffusion Training 

In this repo, we finetune the original Difix3D for the KITTI360 Dataset for Diff-VolumeFusion.


### Step.1 Generating Pseudo-GT Pairs for finetuning.

- (a) Creating degradation images and high quality images pairs using the trained models.
    we default using [input-invariant volumefusion](https://drive.google.com/drive/folders/1sLbprywWeUzXHJkdqplX5rZ3-omQfeFc?usp=sharing) to generated different kinds of degradation levels.

    ```
    cd scripts/evaluations/volumefusion
    cd ../../..
    cd /home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/Validation

    configs_path="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/configs/Models_Lab/VolumeFusion/volumefusion_revision_complete_kitti360.py"
    output_folder="/data1/zliu/KITTI360_Completed/KITTI360_Degradtations_Pairs"
    train_filelist="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/filenames/kitti360/train_complete/train.txt"
    ablation_type="NMRFStereo" # "MetricV2" or "NMRFStereo"
    dataset_type="First_LiDAR_3_Uniform"
    pretrained_model_path="/data1/zliu/KITTI360_Completed/FeedForward_3DGS_Performances/KITTI_Complete_112_544/VolumeFusion/checkpoint-159000/"
    iterations=2


    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
    TORCH_USE_CUDA_DSA=0 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config-file accelerate_config_singleGPU.yaml volumeufison/rendered_views_for_diffix3d_training.py \
        --config_path  $configs_path \
        --output_folder $output_folder \
        --train_filelist $train_filelist \
        --ablation_type $ablation_type \
        --dataset_type $dataset_type \
        --pretrained_model_path $pretrained_model_path \
        --iterations $iterations

    ```
- (b) Create the `train` and the `test` split to training the difix3D SD-Turbo model.

    ```
    cd preprocessing/training_validation_split
    python difix_3d_trainval_split.py
    ```

### Step.2 Finetuning the Pix2Pix SD-Turbo
(1) Finetuning with the weight of the [nvidia/difix_ref](https://huggingface.co/nvidia/difix_ref). 

```
cd codes/Difix3D/scripts
sh train_difix_ref.sh
```


(2) Finetuning with the weight of the [nvidia/difix](nvidia/difix)

```
cd codes/Difix3D/scripts
sh train_difix_no_ref.sh
```


### Step.3 Inference with the Pre-trained Models.

(1) Inference with pre-trained `nvidia/difix_ref`
```
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

```

(2) Inference with pre-trained `nvidia/difix`


```
RUN_THE_DIFIX_WITHOUT_REF_INFRENCE(){
cd ..

model_name="nvidia/difix"
model_path="/data4/zliu/Difix3D/Pretrained_Models/Fintune_Difix_No_Ref/checkpoints/model_1.pkl"
input_image_path="/data4/zliu/Difix3D/KITTI360_Degradtations_Pairs/image/0/scene2013_05_28_drive_0000_sync_bin002_19.png"
ref_image_path="/data4/zliu/Difix3D/KITTI360_Degradtations_Pairs/ref_image/0/scene2013_05_28_drive_0000_sync_bin002_19.png"
prompt="remove degradation"
output_dir="/home/zliu/Project2025/OneStageTraining/FeedStereoGS/codes/Difix3D/outputs/demo_output"
timestep=399
guidance_scale=5.0
ablation_name="official_difix"
use_weight_format="ckpt"

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
```