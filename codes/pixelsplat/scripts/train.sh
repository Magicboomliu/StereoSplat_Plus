TRAIN_THE_MODEL(){
cd ..
python3 -m src.main +experiment=re10k data_loader.train.batch_size=1
}



TRAIN_THE_MODEL