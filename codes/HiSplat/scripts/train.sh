
Train_HiSplat(){
cd ..

python -m src.main +experiment=re10k data_loader.train.batch_size=2 device=auto output_dir=EXP_SAVING_PATH trainer.val_check_interval=3000

}


Train_HiSplat