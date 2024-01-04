# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 01:56:58 2023

@author: Edoardo Giancarli
"""

#### image simulation
import os
os.chdir("D:/Home/File/Python_Project/App_GUI_DL/")

import image_simulation as sim


## train dataset

#!!! %matplotlib inline
train_imgs = "D:/Home/File/Python_Project/App_GUI_DL/Images/Img_Train/"

# imgs
for fig in ['triangle', 'square', 'circle']:
    sim.SimulateImages().image_simulation(num_imgs=100, figure=fig, dim=32, noise=True, plot=False, directory=train_imgs, _n=1334)


## test dataset

#!!! %matplotlib inline
test_imgs = "D:/Home/File/Python_Project/App_GUI_DL/Images/Img_Test/"

# imgs
for fig in ['triangle', 'square', 'circle']:
    sim.SimulateImages().image_simulation(num_imgs=100, figure=fig, dim=32, noise=True, plot=False, directory=test_imgs, _n=1000)


# end