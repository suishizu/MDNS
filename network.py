import torch
import numpy as np
import torch.nn as nn
from networks.Backbone import my_resnet
from networks.MultiADNS import MDNS

class Net(nn.Module):
    
    def __init__(self, snr=None):
        super(DVT, self).__init__()

        num_class = 7

        self.s_backbone_1 = my_resnet(is_Double=False)
        self.s_backbone_2 = my_resnet(is_Double=True)
        self.s_backbone_3 = my_resnet(is_Double=False)

        self.ADNS = ADNS(num_class=num_class)

    def forward(self, fft, stft, real):
        a = self.s_backbone_1(fft)
        b = self.s_backbone_2(stft)
        c = self.s_backbone_3(real)

        output = self.MDNS(a, b, c)

        return output
    
    def get_parameter_groups(self): 
        backbone_params = [
            {'params': self.s_backbone_1.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': self.s_backbone_2.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4}, 
            {'params': self.s_backbone_3.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
  
        ]

        model_params = [
            {'params': self.MDNS.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4} 
        ]

        return backbone_params + model_params
    






