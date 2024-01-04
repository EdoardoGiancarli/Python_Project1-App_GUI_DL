# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:12:21 2023

@author: Edoardo Giancarli
"""

#### libraries

import torch
import torch.nn as nn
from tqdm import tqdm

#### content

# _WarmUpGpu (class)


class _WarmUpGpu(nn.Module):
    """
    Simple 1D CNN model to warm-up the gpu.
    """
    
    def __init__(self):
        super(_WarmUpGpu, self).__init__()
        
        # set up layers
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2), nn.PReLU(),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=5, padding=2), nn.PReLU(),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=5, padding=2))
    
    def forward(self, x):
        x = self.layers(x)
        
        return x
    
    def warm_up(self):
        model = _WarmUpGpu()
        
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Training on GPU...")
            model = model.to(device)
        else:
            raise ValueError("GPU not available.")
        
        for i in tqdm(range(5000)):
            inputs = torch.randn(20, 1, 100).to(device)   # (batch_size, input_channels, input_length)
            targets = torch.randn(20, 1, 100).to(device)  # (batch_size, output_channels, output_length)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
    
        print("GPU warmed up!")


# end