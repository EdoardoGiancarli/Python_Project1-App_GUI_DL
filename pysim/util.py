# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:15:38 2023

@author: Edoardo Giancarli
"""

#### libraries

import torch
from torch import Tensor
import torch.nn as nn

#### content

# _Util (class)
#
# _Normalize (class)


class Util:
    """
    Class with modules with customizable Convolution kernels and Batch Normalisation operations.
    """

    def conv_kernels(in_channels: int,
                     out_channels: int,
                     kernel_size: int,
                     stride: int = 1,
                     padding: int = 0,
                     groups: int = 1,
                     bias: bool = True,
                     bias_zero_init: bool = True,
                     Gauss_kernel: bool = True,
                     transpose: bool = False) -> nn.Module:
        
        """
        Setting convolution operations with xavier initialisation kernels.
        ---------------------------------------------------------------------------
        Ref:
            [1] X. Glorot, Y. Bengio, "Understanding the difficulty of training deep
                feedforward neural networks" (2010)
        """
        
        if not transpose:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                             groups=groups, bias=bias)
        else:
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                      padding, groups=groups, bias=bias)
        
        if Gauss_kernel:
            nn.init.xavier_normal_(conv.weight)
        else:
            nn.init.xavier_uniform_(conv.weight)
        
        if bias and bias_zero_init:
            conv.bias.data.fill_(0)
        
        return conv
    
    
    def bn(num_features: int,
           eps: float = 1e-5,
           momentum: float = 0.1,
           zero_init: bool = False) -> nn.Module:
        
        """
        Setting batch normalisation gamma parameters to zero.
        ---------------------------------------------------------------------------
        Ref:
            [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep
                Network Training by Reducing Internal Covariate Shift" (2015)
            [2] T. He et al., "Bag of Tricks for Image Classification with
                Convolutional Neural Networks" (2019)
        """
        
        bn = nn.BatchNorm2d(num_features, eps, momentum)
        
        # init gamma to zero (default = 1)
        if zero_init:
            bn.weight.data.fill_(0)
        
        return bn



class Normalize(object):
    """
    Images normalisation wrt mean and std values + normalisation
    between [0, 1] or [-1, 1].
    ------------------------------------------------------------
    Par:
        centering (bool): centering of input tensor to mean = 0 and std = 1
                          (default = True)
        norm_range (str): normalisation range, 'unilateral' for [0, 1],
                          'bilateral' for [-1, 1] (default = 'unilateral')
                                          
    Ref:
        [1] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
        [2] Ivezić, Ž., Connolly, A. J., VanderPlas, J. T., & Gray, A., "Statistics, data mining, and
            machine learning in astronomy: a practical Python guide for the analysis of survey data" (2020)
    """
    
    def __init__(self, centering: bool = True,
                 norm_range: str = 'unilateral'):
        
        self.center = centering
        self.range = norm_range
        
    def __call__(self, tensor: Tensor) -> Tensor:
        
        if self.center:
            tensor = (tensor - torch.mean(tensor))/(torch.std(tensor) + 1e-12)
        
        tensor = (tensor - torch.min(tensor))/(torch.max(tensor) - torch.min(tensor))
        
        if self.range == 'unilateral':
            pass
        elif self.range == 'bilateral':
            tensor = 2*tensor - 1
        else:
            raise ValueError("norm_range must be 'unilateral' for a norm. range in [0, 1]",
                             "or 'bilateral' for a norm. range in [-1, 1]")
        
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


# end