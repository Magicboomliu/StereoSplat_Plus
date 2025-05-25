from mmengine.config import Config

import sys
sys.path.append("..")
from depthsplat.models.encoder import get_encoder



if __name__=="__main__":
    
    cfg = Config.fromfile('../configs/DepthSplat/depthsplat_gs_kitti360_stereo_224x840.py')
    
    
    
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
