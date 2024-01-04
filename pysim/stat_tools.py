# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:37:13 2023

@author: Edoardo Giancarli
"""

#### libraries

import pathlib
import os
from PIL import Image
from typing import Union
from tqdm import tqdm
import numpy as np

import torch
from torch.nn import Module, Softmax
from torchvision import transforms

import matplotlib.pyplot as plt
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=30, usetex=True)

from .simple_tools import SimpleTools
smt = SimpleTools()

#### content

# StatTools (class)


class StatTools:
    """
    Analysis of SimpleNet classifying performance.
    ---------------------------------------------------------------------------
    Methods:
        
    """
    
    def _take_images(self, dir_path):
        
        imgdir_path = pathlib.Path(dir_path)
        file_list = sorted([str(path) for path in imgdir_path.glob('*.png')])
        
        return file_list
            
    def _get_data(self, images_list, class_names):
        
        labels = [next(key for key, value in class_names.items() if value
                       in os.path.basename(file)) for file in images_list]
    
        data = [(img, label) for img, label in zip(images_list, labels)]
        
        return data
    
    def verify_model(self, images_path: str,
                     model: [str, Module],
                     class_names: dict,
                     transform: transforms = None) -> Union[list[int], list[int]]:
        
        """
        Verify the performance of the model.
        -------------------------------------------------------------------
        Par:
            images_path (str): path for the images
            model (Module): CNN model
            class_names (dict): dictionary with ground truth labels for classes
            transform (torchvision.transforms): transformation to apply to the images (default = None)
            
        Return:
            true_positive_probs (list): probabilities with which the images are correctly
                                        classified by the input model
            false_positive_probs (list): probabilities with which the images are wrongly
                                         classified by the input model
        """
        
        img_list = self._take_images(images_path)
        
        data = self._get_data(img_list, class_names)
        
        true_positive_probs = []
        false_positive_probs = []
        
        if isinstance(model, str):
            model, _ = SimpleTools().load_model(model)
        
        for file in tqdm(data):
            img = Image.open(file[0]).convert("RGB")
            
            if transform is not None:
                img = transform(img)
            else:
                img = transforms.ToTensor()(img)
            
            model_output = model(torch.unsqueeze(img, axis=0))
            probabilities = Softmax(dim=1)(model_output)
            top_prob = torch.max(probabilities).item()
            
            if torch.argmax(probabilities, dim=1).item() == file[1]:
                true_positive_probs.append(round(top_prob*100))
            else:
                false_positive_probs.append(round(top_prob*100))
            
        return true_positive_probs, false_positive_probs
    
    def probs_histogram(self, images_path: str,
                        model: [str, Module],
                        class_names: dict,
                        transform: transforms = None,
                        N_bins: int = 10,
                        density: bool = False):
        
        """
        Probabilities histogram for the correctly or wrongly classified images.
        -------------------------------------------------------------------
        Par:
            images_path (str): path for the images
            model (Module): CNN model
            class_names (dict): dictionary with ground truth labels for classes
            transform (transforms): transformation to apply to the images (default = None)
            N_bins (int): number of bins for the histogram (default = 10)
            density (bool): data PDF (default = False)
        """
        
        probs = self.verify_model(images_path, model, class_names, transform)
        
        bin_edges = np.linspace(min(min(probs[0]), min(probs[1])), max(max(probs[0]), max(probs[1])), N_bins)
        
        fig = plt.figure(num=None, figsize=(12, 12), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.hist(probs[0], bins=bin_edges, density=density, color='Lime', alpha=0.4, label='True positive')
        ax.hist(probs[1], bins=bin_edges, density=density, color='OrangeRed', alpha=0.6, label='False positive')
        plt.title("Classification probs histogram")
        plt.xlabel('classification probability (in $\%$)')
        plt.legend(loc='best')
        
        if not density:
            plt.ylabel('N of images')
        else:
            plt.ylabel('images PDF')
        
        ax.grid(alpha=0.5)
        ax.label_outer()            
        ax.tick_params(which='both', direction='in',width=2)
        ax.tick_params(which='major', direction='in',length=7)
        ax.tick_params(which='minor', direction='in',length=4)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.show()


# end