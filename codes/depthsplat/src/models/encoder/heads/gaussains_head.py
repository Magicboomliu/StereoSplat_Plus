import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import sys
from ..unimatch.dpt_head import DPTHead


# Gaussain Estimation Head
class Gaussains_Estimator_Head(nn.Module):
    def __init__(self,
                 monodepth_vit_type,
                 upsample_factor,
                 num_scales):
        super().__init__()
        
        self.monodepth_vit_type = monodepth_vit_type
        self.upsample_factor = upsample_factor
        
        # upsample features to the original resolution
        model_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 96, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 128, 'out_channels': [128, 256, 512, 1024]},}
        
        self.feature_upsampler = DPTHead(**model_configs[monodepth_vit_type],
                                        downsample_factor=upsample_factor,
                                        return_feature=True,
                                        num_scales=num_scales,)
        
        feature_upsampler_channels = model_configs[monodepth_vit_type]["features"]



    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def dtype(self):
        return next(self.parameters()).dtype




    def forward(self):
        pass


if __name__=="__main__":
    pass