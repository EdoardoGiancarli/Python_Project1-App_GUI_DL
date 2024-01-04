# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:00:39 2023

@author: Edoardo Giancarli
"""

#### libraries

from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from typing import Union

from .simple_tools import SimpleTools

#### content

# apply_model (function)


def apply_model(img_path: str,
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
        model, _ = SimpleTools().load_model(model)
    
    model_output = model(torch.unsqueeze(img, axis=0))
    
    probs = nn.Softmax(dim=1)(model_output)
    top_prob = torch.max(probs).item()
    top_pred = class_names[torch.argmax(probs, dim=1).item()]
    
    print("This is a " + top_pred + " with a probability of: " + str(round(top_prob*100)) + "%")
    
    return top_pred, top_prob


# end