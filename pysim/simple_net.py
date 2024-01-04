# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:33:09 2023

@author: Edoardo Giancarli
"""

#### libraries

from torch import Tensor
import torch.nn as nn

from .util import Util as ut

#### content

# SimpleNet (class)


class SimpleNet(nn.Module):
    """
    Base architecture for the SimpleNet CNN model.
    ---------------------------------------------------------------------------
    Parameters:
        act_func (str): select the activation funtion used in the residual
                        block ('ReLU' or 'PReLU', default = PReLU)
        
    Architecture:
        1. Conv, BatchNorm, PReLU (or ReLU)
        2. Conv, BatchNorm, PReLU (or ReLU) + Dropout
        3. Conv, BatchNorm, PReLU (or ReLU) + MaxPool
        4. Flatten
        5. linear + PReLU (or ReLU) + Dropout
        6. linear out + softmax
    
    Ref:
        [1] K. He et al., "Deep Residual Learning for Image Recognition" (2015)
        [2] K. He et al., "Identity Mappings in Deep Residual Networks" (2016)
        [3] H. Ren et al., "DN-ResNet: Efficient Deep Residual Network for Image Denoising" (2018)
        [4] S. Xie et al., "Aggregated Residual Transformations for Deep Neural Networks" (2017)
        [5] T. He et al., "Bag of Tricks for Image Classification with Convolutional Neural Networks" (2019)
    """
    
    def __init__(self, act_func: str = 'PReLU') -> nn.Module:
        
        super(SimpleNet, self).__init__()
        
        # set up convolution with Gaussian kernels
        conv1 = ut.conv_kernels(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        conv2 = ut.conv_kernels(in_channels=16, out_channels=32, kernel_size=5, padding=2)  
        conv3 = ut.conv_kernels(in_channels=32, out_channels=1, kernel_size=5, padding=2)
        
        # set up BNs
        bn1 = ut.bn(num_features=16, eps=1e-6, momentum=0.1, zero_init=False)
        bn2 = ut.bn(num_features=32, eps=1e-6, momentum=0.1, zero_init=False)
        bn3 = ut.bn(num_features=1, eps=1e-6, momentum=0.1, zero_init=False)
        
        # set up first two layers (fixed)
        self.architecture = nn.Sequential(
            conv1, bn1, self._act_func(act_func),
            conv2, bn2, self._act_func(act_func), nn.Dropout(p=0.5),
            conv3, bn3, self._act_func(act_func), nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=324, out_features=256), self._act_func(act_func), nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=3))
    
    def forward(self, x) -> Tensor:
        
        x = self.architecture(x)
        
        return x
    
    def _act_func(self, activation : str) -> str:
        
        if activation == 'PReLU':
            act = nn.PReLU()
        elif activation == 'ReLU':
            act = nn.ReLU()
        else:
            raise ValueError("You must choose the activation function for the res. block: 'ReLU' or 'PReLU'")
        
        return act


# end
