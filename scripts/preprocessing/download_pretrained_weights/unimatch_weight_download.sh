
Download_Unimatch_and_DepthAnything(){

pretrained_weight_locations="/data1/zliu/pretrained_foundataion_models/depth_estimation/Depthsplat/"

wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth -P $pretrained_weight_locations
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -P $pretrained_weight_locations


}

Download_Unimatch_and_DepthAnything