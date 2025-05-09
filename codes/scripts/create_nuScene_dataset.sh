Create_The_NuScene_Dataset(){

cd ..
cd preprocessing

root_path="/data2/zliu_backup_data/nuScenes/v1.0-mini/"
version="v1.0-mini"
min_bin_length=3.2
out_dir="/data1/zliu/FeedForwardGS_Datasest/nuscence/small_val/"


python nuScene_bin_creation.py nuscenes \
                    --root-path $root_path  \
                    --version $version \
                    --min_bin_length $min_bin_length \
                    --out-dir $out_dir

}


Create_The_NuScene_Dataset