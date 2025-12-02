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
sh train_difix_re.sh
```