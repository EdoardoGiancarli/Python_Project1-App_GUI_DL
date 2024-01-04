# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 22:30:10 2023

@author: Edoardo Giancarli

References:
    1. Python Simplified - Machine Learning GUI App (url: https://github.com/MariyaSha/ml_gui_app/tree/main)
"""

###############  GUI app for DL - version 1  ###############

####   libraries   ####

import os
os.chdir("D:/Home/File/Python_Project/App_GUI_DL/")    # path with modules

from taipy.gui import Gui
import pysim as psm
from torchvision import transforms

from PIL import Image
import torch
import torch.nn as nn
from typing import Union

####    content    ####

# GUI app

####     codes     ####

# model output classes
class_names = {0: 'triangle',
               1: 'square',
               2: 'circle'}

# transform images
transform = transforms.Compose([transforms.functional.to_grayscale,
                                transforms.ToTensor(),
                                transforms.Resize((36, 36), antialias=True),
                                psm.Normalize()])

# model
model = 'SimpleNet_model1_1.pth'

def apply(img_path: str,
          model: [str, nn.Module],
          class_names: dict,
          transform: transforms = None) -> Union[str, float]:
    
    """
    Model application on the chosen image.
    """
    
    # open and convert image
    img = Image.open(img_path).convert("RGB")
    
    # transform image for the model
    if transform is not None:
        img = transform(img)
    else:
        img = transforms.ToTensor()(img)
    
    # model application
    if isinstance(model, str):
        model, _ = psm.SimpleTools().load_model(model)
    
    model_output = model(torch.unsqueeze(img, axis=0))
    
    probs = nn.Softmax(dim=1)(model_output)
    top_prob = torch.max(probs).item()
    top_pred = class_names[torch.argmax(probs, dim=1).item()]
    
    # print("This is a " + top_pred + " with a probability of: " + str(int(top_prob*100)) + "%")
    
    return top_pred, top_prob




content = ""
img_path = "D:/Home/File/Python_Project/App_GUI_DL/Images/placeholder_image.png"
prob = 0
pred = ""

index = """
<|text-center|
<|{"D:/Home/File/Python_Project/App_GUI_DL/Images/logo.png"}|image|width=25vw|>

<|{content}|file_selector|extensions=.png|>
select an image from your file system

<|{pred}|>

<|{img_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""

# update GUI
def on_change(state, var_name, var_val):
    if var_name == "content":
        top_pred, top_prob = apply(img_path, model, class_names, transform)
        state.prob = round(top_prob * 100)
        state.pred = "this is a " + top_pred + "!"
        state.img_path = var_val


app = Gui(page=index)


if __name__ == "__main__":
    app.run(use_reloader=True)


# end