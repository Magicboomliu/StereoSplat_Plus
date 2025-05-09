
TRAIN_The_OmniScence(){

CUDA_VISIBLE_DEVICES=1,2 accelerate launch train.py \
        --py-config configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py \
        --work-dir workdirs/omni_gs_nusc_novelview_r50_224x400 
}


TRAIN_The_OmniScence