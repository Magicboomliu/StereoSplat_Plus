import sys
sys.path.append("..")
from data.KITTI360.dataloader import KITTI360Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def check_dict(dict_info):
    invalid= False
    for key in dict_info.keys():
        if check_tensor_for_nan_inf(dict_info[key]):
            invalid = True
    
    return invalid
        

def check_tensor_for_nan_inf(tensor):
    if isinstance(tensor, torch.Tensor):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return True
    return False


if __name__=="__main__":
    dataset_params = {
        "datapath":"/data1/StereoDatasets/KITTI/KITTI360/",
        "train_filelist":"/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/all_2013_05_28_drive_0000_sync.txt",
        "val_filelist":"/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt",
        "test_filelist":"/home/zliu/Project2025/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync.txt",
        "data_version":"bin_infos_8.0",
        "resolution":[224, 840], # idx 0 is the proceseed image resolution, the last is the the initial image resolution
        "split":"train",
        "sequence":'2013_05_28_drive_0000_sync',
        "use_center":True,
        "use_first": False,
        "use_last": False,
    }
    
    train_dataset = KITTI360Dataset(**dataset_params)
    


    train_dataloader = DataLoader(
        train_dataset, 1, shuffle=False,
        num_workers=0
    )
    
    for batch in tqdm(train_dataloader):
        # dict_keys(['bin_token', 'outputs', 'inputs', 'inputs_pix', 'inputs_vol'])
        # print(batch['bin_token']) # ['scene2013_05_28_drive_0000_sync_bin000.pkl']

        output_state = check_dict(batch['outputs'])
        input_state = check_dict(batch['inputs'])
        input_pix_state = check_dict(batch['inputs_pix'])
        input_vol_state = check_dict(batch['inputs_vol'])
        
        
        state = output_state + input_state + input_pix_state +input_vol_state
        
        if state:
            print(batch['bin_token'])
            print("output state: ",output_state)
            print("input state: ",input_state)
            print("input pix state: ",input_pix_state)
            print("input vol state: ",input_vol_state)

    print("finished")