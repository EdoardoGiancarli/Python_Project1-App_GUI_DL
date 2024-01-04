# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:25:38 2023

@author: Edoardo Giancarli
"""

#### libraries

# from tqdm import tqdm

# import torch
# from torch.nn import Module
# from torch.utils.data import DataLoader

# from .simple_tools import SimpleTools

#### content

# MeanModel (class)
#
# TrainMultipleModels (class)


class MeanModel:
    """
    Average different version of the DL model for better generalizability.
    ---------------------------------------------------------------------------
    Attributes:
        

    Methods:
        
    """
    pass


# class _ModelAnalysis:
#     """
#     Analysis of the SimpleNet model.
#     """
    
#     def __init__(self, model: Module,
#                  epochs: [int, list],
#                  learning_rate: [float, list],
#                  train_dataset: DataLoader,
#                  test_dataset: DataLoader):
    
#         self.model = model
        
#         self.epochs = self._verify_obj_list(epochs)
#         self.learning_rate = self._verify_obj_list(learning_rate)
        
#         self.train_dataset = train_dataset
#         self.test_dataset = test_dataset
        
#         self.smt = SimpleTools()
        
#     ##########################################################################################
    
#     def _verify_obj_list(self, obj):
        
#         if isinstance(obj, list):
#             return obj
#         else:
#             return list(obj)
    
    
#     def _train(self):
        
#         # initialize loss and accuracy
#         train_loss, train_accuracy = [], []
        
#         torch.cuda.empty_cache()
        
#         # get train (mean) loss and accuracy values
#         for e, lr in zip(self.epochs, self.learning_rate):
#             _, tl, ta, _, _ = self.smt.train_model(self.model, epochs=e, learn_rate=lr,
#                                                    train_dataset=self.train_dataset)
            
#             train_loss += tl
#             train_accuracy += ta
        
#         return train_loss, train_accuracy
    
    
#     def _test(self):
        
#         # get test (mean) accuracy value
#         test_accuracy = self.smt.test_model(self.test_dataset)
        
#         return test_accuracy
    
    
#     def _save(self, batch_size, epochs, lr, activation,
#               model_ID, train_loss, train_accuracy):
        
#         self.smt.save_model(batch_size, epochs, lr, activation,
#                             model_ID, train_loss, train_accuracy)
    
#     ##########################################################################################
    
#     def model_analysis(self, batch_size: int,
#                        activation: str,
#                        model_ID: str):
        
#         """
#         Performs the model analysis (train, test, saving).
#         """
        
#         train_loss, train_accuracy = self._train()
        
#         print(len(train_loss), len(train_accuracy))
        
#         self.smt.show_model(train_loss, train_accuracy)
        
#         self._test()
        
#         self._save(batch_size, self.epochs, self.learning_rate, activation,
#                    model_ID, train_loss, train_accuracy)
    


# def train_multiple_models(model: Module,
#                           n: int,
#                           epochs: [int, list],
#                           learning_rate: [float, list],
#                           train_dataset: DataLoader,
#                           test_dataset: DataLoader,
#                           batch_size: int,
#                           activation: str,
#                           model_ID: str):
    
#     """
#     Train the model multiple times.
#     """
    
#     for _ in tqdm(range(n)):
        
#         mas = _ModelAnalysis(model, epochs, learning_rate,
#                              train_dataset, test_dataset)
        
#         mas.model_analysis(batch_size, activation, model_ID + '_trial' + str(n))
        
#         del mas.model, mas.smt.model


# end