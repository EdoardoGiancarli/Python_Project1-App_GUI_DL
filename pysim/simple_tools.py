# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:37:13 2023

@author: Edoardo Giancarli
"""

#### libraries

import numpy as np                              # operations
# import pandas as pd                             # dataframe 
import random
import pathlib                                  # filepaths
import os                    
from PIL import Image                           # images
from tqdm import tqdm                           # loop progress bar

import torch                                    # pytorch 
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

import subprocess                               # GPU memory check
from typing import Union                        # hints

import matplotlib.pyplot as plt                 # plotting
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=30, usetex=True)

# import pdb                                      # for debugging 
# pdb.set_trace()

#### content

# ImageDataset (class)
#
# SimpleTools (class)


class _ImageDataset(Dataset):
    """
    Features and targets coupling.
    ---------------------------------------------------------------------------
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
    """
    
    def __init__(self, file_list: list,
                 labels: list,
                 transform: transforms = None):
        
        self.labels = labels
        
        if transform is not None:
            self.tensor_list = [transform(Image.open(img).convert("RGB")) for img in tqdm(file_list)]
        else:
            self.tensor_list = [Image.open(img).convert("RGB") for img in tqdm(file_list)]
    
    
    def __getitem__(self, index):
        
        tensor = self.tensor_list[index]
        label = self.labels[index]
        
        return tensor, label
    
    
    def __len__(self):
        return len(self.labels)


class SimpleTools:
    """
    This class contains the functions to train the CNN model, to plot the loss and the
    accuracy of the model, to test the CNN and to save/load the trained model.
    ---------------------------------------------------------------------------
    Attributes:
        model (nn.Module): CNN model (in train_model module)
        loss_fn (nn.Module): loss for the training (Mean Squared Error Loss, in train_model module)
        optimizer (torch.optim): features optimizer (Adam, in train_model module)
        device (torch): device on which the computation is done (in train_model module)

    Methods:
        make_dataset: it defines the train datasets
        train_model: it trains the CNN model
        cascade_training: it performs the cascade learning of the model
        show_model: it shows the loss and the accuracy of the trained CNN model
        test_model: it tests the CNN model after the training
        save_model: it saves the CNN model (or if you want to save a checkpoint during training)
        load_model: it loads the saved CNN model (or the checkpoint to continue the CNN training)
        manage_dataset: it saves or loads a dataset
        
    Ref:
        [1] S. Raschka, Y. H. Liu, V. Mirjalili "Machine Learning with PyTorch
            and Scikit-Learn" (2022)
        [2] F. Fleuret, Deep Learning Course 14x050, University of Geneva,
            url: https://fleuret.org/dlc/
        [3] H. Ren et al., "DN-ResNet: Efficient Deep Residual Network for Image Denoising" (2018)
        [4] DP Kingma and J. Ba, "Adam: A Method for Stochastic Optimization" (2014)
    """
    ##########################################################################################
    
    def _gpu_memory_nvidia_smi(self) -> str:
        
        try:
            # nvidia-smi command to get GPU memory usage
            result = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
    
            # search for memory usage information in the output
            memory_usage_info = ""
            for line in result.splitlines():
                if "MiB / " in line:
                    memory_usage_info = line.strip()
                    break
    
            if memory_usage_info:
                print("\n GPU Memory Usage:", memory_usage_info)
            else:
                print("\n Unable to find GPU memory usage information.")
        
        except subprocess.CalledProcessError:
            print("\n Error running nvidia-smi command. Make sure it is installed and accessible.")
    
    
    def _plot(self, x, y, label, xlabel, ylabel, title, ylim=None,
              y2=None, color2=None, label2=None):
        
        fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)
        ax = fig.add_subplot(111)
        ax.plot(x, y, label=label)
        
        if y2 is not None:
            ax.scatter(x, y2, c=color2, label=label2)
        
        plt.xlim((0.5, len(y) + 0.5))
        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)            
        plt.legend(loc = 'best')
        ax.grid(True)
        ax.label_outer()            
        ax.tick_params(which='both', direction='in',width=2)
        ax.tick_params(which='major', direction='in',length=7)
        ax.tick_params(which='minor', direction='in',length=4)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.show()
    
    ##########################################################################################
    
    
    def make_dataset(self, data_path: str,
                     batch_size: int,
                     class_names: dict,
                     valid_size: float = None,
                     transform: transforms = None) -> DataLoader:
        
        """
        Train dataset generation (and also validation dataset if valid_size is inserted).
        -------------------------------------------------------------------
        Par:
            data_path (str): path for the data
            batch_size (int): batch size for the train (and validation) dataset
            class_names (dict): dictionary with ground truth labels for classes
            transform (torchvision.transforms): transformation to apply to the images (default = None)
            valid_size (int): validation dataset size in percentage wrt full size;
                              valid_size in [0, 1) (default = None)
            
        Return:
            train_dataset (DataLoader): train dataset
            valid_dataset (DataLoader): validation dataset (if valid_size is defined)
        """
        
        # load images
        imgdir_path = pathlib.Path(data_path)
        file_list = sorted([str(path) for path in imgdir_path.glob('*.png')])
        for _ in range(10):
            random.shuffle(file_list)

        # define labels
        labels = [next(key for key, value in class_names.items() if value
                       in os.path.basename(file)) for file in file_list]
        
        # create the dataset
        print("Making the dataset...")
        image_dataset = _ImageDataset(file_list, labels, transform=transform)
        
        # split and define train and validation dataset
        if valid_size is not None:
            
            if valid_size < 0 or valid_size >= 1:
                raise ValueError("valid_size must assume a value in the range [0, 1).")
            
            v = int(valid_size*len(image_dataset))
            validation = Subset(image_dataset, torch.arange(v))
            training = Subset(image_dataset, torch.arange(v, len(image_dataset)))
            
            train_dataset = DataLoader(training, batch_size, shuffle=True)
            valid_dataset = DataLoader(validation, batch_size, shuffle=False)
            
            return train_dataset, valid_dataset
        
        else:
            train_dataset = DataLoader(image_dataset, batch_size, shuffle=True)
            return train_dataset
        
        
    def train_model(self, model: object,
                    epochs: int,
                    learn_rate: float,
                    train_dataset: DataLoader,
                    valid_dataset: DataLoader = None) -> Union[nn.Module, list, list, list, list]:
        
        """
        Training of the CNN model defined in DnsResNet.
        ------------------------------------------------------
        Par:
            model (nn.Module): CNN model
            epochs (int): number of iterations for the model training
            learn_rate (float): learning rate parameter for the model optimization
            train_dataset (DataLoader): training dataset
            valid_dataset (DataLoader): validation dataset (default = None)
        
        Return:
            model (nn.Module): trained model
            mean_loss_train (list): mean loss values for the model training
            mean_accuracy_train (list): mean accuracy values for the model training
            mean_loss_valid (list): mean loss values for the model validation (if valid_dataset is inserted)
            mean_accuracy_valid (list): mean accuracy values for the model validation (if valid_dataset is inserted)
        """
        
        # set model and stage as global variables
        self.model = model
        
        # define loss and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=0, amsgrad=False)
        
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learn_rate, momentum=0,
        #                                  dampening=0, weight_decay=0, nesterov=False)
        
        print_every = epochs/30
        
        # control device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Training on GPU...")
            self.model = self.model.to(self.device)
        
        else:
            print("No GPU available, redirecting to CPU...\n")
            user_input = input("Continue training on CPU? (y/n): ")
            
            if user_input.lower() == "n":
                raise Exception("Training interrupted")
            else:
                self.device = torch.device("cpu")
        
        # define lists for loss and accuracy (both train and validation)
        self.mean_loss_train = [0]*epochs
        self.mean_accuracy_train = [0]*epochs
        
        if valid_dataset is not None:
            self.mean_loss_valid = [0]*epochs
            self.mean_accuracy_valid = [0]*epochs
        else:
            self.mean_loss_valid = None
            self.mean_accuracy_valid = None
        
        # reduce memory cost by mixing the precision of float data
        scaler = GradScaler()
        
        # training loop
        for epoch in tqdm(range(epochs)):
            
            # model training
            self.model.train()
            
            for x_batch, y_batch in train_dataset:
                x_batch = x_batch.to(self.device) 
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()                                          # put the optimizer grad to zero
                
                # mixed precision for float                
                with autocast():
                    pred = self.model(x_batch)                                      # model prediction
                    loss = self.loss_fn(pred, y_batch)                              # model loss
                
                scaler.scale(loss).backward()                                       # backward propagation 
                scaler.step(self.optimizer)                                         # model parameters optimization
                scaler.update()
                
                self.mean_loss_train[epoch] += loss.item()*y_batch.size(0)                  # store single loss values 
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float().cpu().numpy()
                self.mean_accuracy_train[epoch] += is_correct.sum()                         # store single accuracy values

            self.mean_loss_train[epoch] /= len(train_dataset.dataset)               # mean loss value for the epoch
            self.mean_accuracy_train[epoch] /= len(train_dataset.dataset)           # mean accuracy value for the epoch
            
            if int(print_every) >= 1 and epoch % int(print_every) == 1:
                print("####################\n",
                      f"Training Loss: {self.mean_loss_train[epoch]:.4f}")
            
            # model validation
            if valid_dataset is not None:
            
                self.model.eval()
                
                with torch.no_grad():
                    for x_batch, y_batch in valid_dataset:
                        x_batch = x_batch.to(self.device) 
                        y_batch = y_batch.to(self.device)
                        
                        valid_pred = self.model(x_batch)                                                 # model prediction for validation
                        valid_loss = self.loss_fn(valid_pred, y_batch)                                   # model loss for validation
                        self.mean_loss_valid[epoch] += valid_loss.item()*y_batch.size(0)                 # store single validation loss values
                        is_correct = (torch.argmax(valid_pred, dim=1) == y_batch).float().cpu().numpy() 
                        self.mean_accuracy_valid[epoch] += is_correct.sum()                              # store single validation accuracy values
    
                self.mean_loss_valid[epoch] /= len(valid_dataset.dataset)                  # validation mean loss value for the epoch
                self.mean_accuracy_valid[epoch] /= len(valid_dataset.dataset)              # validation mean accuracy value for the epoch
                
                if int(print_every) >= 1 and epoch % int(print_every) == 1:
                    print(f"Validation Loss: {self.mean_loss_valid[epoch]:.4f}")
        
        # check on GPU memory
        self._gpu_memory_nvidia_smi()
        
        # return output
        return self.model, self.mean_loss_train, self.mean_accuracy_train, self.mean_loss_valid, self.mean_accuracy_valid
    
    
    def show_model(self, train_loss: list = None,
                   train_accuracy: list = None,
                   valid_loss: list = None,
                   valid_accuracy: list = None,
                   title_notes: str = None,
                   comp_dloss: bool = False):
        
        """
        Plots of the trained CNN model loss and accuracy (also with validation if
        valid_dataset in train_model() is defined).
        ------------------------------------------------------
        Par:
            train_loss (list, array): show a specific train loss (default = None)
            train_accuracy (list, array): show a specific train accuracy (default = None)
            valid_loss (list, array): show a specific validation loss (default = None)
            valid_accuracy (list, array): show a specific validation accuracy (default = None)
            title_notes (str): notes to add to the plot title (default = None)
            comp_dloss (bool): if True computes an approximate derivative for the train loss (default = False)
        """
        
        # define losses
        if train_loss is not None and valid_loss is not None:
            tl, ta = train_loss, train_accuracy
            vl, va = valid_loss, valid_accuracy
        else:
            tl, ta = self.mean_loss_train, self.mean_accuracy_train
            vl, va = self.mean_loss_valid, self.mean_accuracy_valid
        
        # define ascissa
        x_arr = np.arange(len(tl)) + 1
        
        # define plot title
        title_loss = 'Model mean loss'
        title_accuracy = 'Model mean accuracy'
        if title_notes is not None and isinstance(title_notes, str):
            title_loss += ' ' + title_notes
            title_accuracy += ' ' + title_notes
        
        # loss plot
        self._plot(x_arr, tl, 'train loss', 'epoch', 'loss', title_loss,
                   y2=vl, color2='OrangeRed', label2='valid. loss')
        
        # accuracy plot
        self._plot(x_arr, ta, 'train acc.', 'epoch', 'accuracy', title_accuracy,
                   y2=va, ylim=(0, 1), color2='OrangeRed', label2='valid. acc.')
        
        # loss derivative
        if comp_dloss:
            dloss = [tl[l + 1] - tl[l]
                     for l in range(len(tl) - 1)]
            
            self._plot(x_arr[:-1], dloss, 'loss derivative', 'epoch',
                       'loss change rate', 'Model loss change rate')
    
    
    def test_model(self, test_dataset: DataLoader,
                   model: nn.Module = None):
        
        """
        Test of the CNN model after the training.
        ------------------------------------------------------
        Par:
            test_dataset (DataLoader): test dataset
            model (torch): CNN model (if None, the used model is the one
                           in the train_model module, default = None)
        """
        
        # waits for all kernels in all streams on a CUDA device to complete 
        torch.cuda.synchronize()
        
        # to cpu (for the test)
        if model is None:
            model = self.model.cpu()
        
        # initialize test loss
        self.mean_accuracy_test = 0
        
        # test CNN
        model.eval()
        
        print("Testing the model...")
        with torch.no_grad():
            for x_batch, y_batch in tqdm(test_dataset):
                test_pred = model(x_batch)
                is_correct = (torch.argmax(test_pred, dim=1) == y_batch).float().numpy()
                self.mean_accuracy_test += is_correct.sum()
        
        # test mean accuracy value
        self.mean_accuracy_test /= len(test_dataset.dataset)
        print('The mean accuracy value for the test dataset is:', self.mean_accuracy_test)
        
        return self.mean_accuracy_test
    
    
    def save_model(self, batch_size: int,
                   epochs: int,
                   learning_rate: float,
                   activation: str,
                   filename: str,
                   train_loss: list = None,
                   train_accuracy: list = None,
                   valid_loss: list = None,
                   valid_accuracy: list = None,
                   notes: str = None,
                   gpu: str = None):
        
        """
        To save the CNN model after training (or a checkpoint during training); this module saves the model
        by creating a dictionary in which the model features are stored, such as the model, the model state
        and other information.
        ------------------------------------------------------
        Par:
            batch_size (int): batch size of the training dataset for the training process
            epochs (int): number of iterations for the model training
            learning_rate (float): learning rate parameter for the model optimization
            activation (str): activation function in the residual blocks
            filename (str): name of the CNN model (.pt or .pth, the filepath where to save the
                            model is defined inside the module)
            train_loss (list, array): save a specific train loss (default = None)
            train_accuracy (list, array): save a specific train accuracy (default = None)
            valid_loss (list, array): save a specific validation loss (default = None)
            valid_accuracy (list, array): save a specific validation accuracy (default = None)
            notes (str): possible notes for model description (default = None)
            gpu (str): GPU memory used (default = None)
        """
        
        filepath = "D:/Home/File/Python_Project/App_GUI_DL/Model_n_Datasets/"
        
        # define losses and accuracy
        if train_loss is not None:
            tl, ta = train_loss, train_accuracy
            vl, va = valid_loss, valid_accuracy
        else:
            tl, ta = self.mean_loss_train, self.mean_accuracy_train
            vl, va = self.mean_loss_valid, self.mean_accuracy_valid
        
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      'epochs': epochs,
                      'learning_rate': learning_rate,
                      'mean_loss_train': tl,
                      'mean_accuracy_train': ta,
                      'mean_loss_valid': vl,
                      'mean_accuracy_valid': va,
                      'device': self.device,
                      'optimizer_state': self.optimizer.state_dict(),
                      'batch_size': batch_size,
                      'activation': activation,
                      'notes': notes,
                      'gpu': gpu}
        
        checkpoint['mean_accuracy_test'] = getattr(self, 'mean_accuracy_test', None)
        
        torch.save(checkpoint, filepath + filename)
    
    
    def load_model(self, filename: str) -> Union[nn.Module, dict]:
        
        """
        To load the CNN model (or a checkpoint to continue the CNN training).
        ------------------------------------------------------
        Par:
            filename (str): name of the CNN model (.pt or .pth, the filepath where to save the
                            model is defined inside the module)
            
        Return:
            model (nn.Module): CNN model
            obs (dict): dictionary with epochs, loss, accuracy and batch size
        """
        
        filepath = "D:/Home/File/Python_Project/App_GUI_DL/Model_n_Datasets/"        
        
        # load the model
        checkpoint = torch.load(filepath + filename)
        
        # define update model
        model = checkpoint['model']
        model.optimizer_state = checkpoint['optimizer_state']
        model.load_state_dict(checkpoint['state_dict'])
        model.device = checkpoint['device']
        model.average_loss = checkpoint['mean_loss_train']
        
        # dict with other info
        obs = {'epochs': checkpoint['epochs'],
               'learning_rate': checkpoint['learning_rate'],
               'batch_size': checkpoint['batch_size'],
               'activation': checkpoint['activation'],
               'notes': checkpoint['notes'],
               'gpu': checkpoint['gpu'],
               'mean_loss_train': checkpoint['mean_loss_train'],
               'mean_loss_valid': checkpoint['mean_loss_valid'],
               'mean_accuracy_train': checkpoint['mean_accuracy_train'],
               'mean_accuracy_valid': checkpoint['mean_accuracy_valid'],
               'mean_accuracy_test': checkpoint['mean_accuracy_test']}
        
        return model, obs
    
    
    def manage_dataset(self, dataset: [DataLoader, str] = None,
                       dataset_name: str = None,
                       mode: str = 'save') -> DataLoader:
        
        """
        Dataset management: save or load a dataset (.pt file).
        ------------------------------------------------------
        Par:
            dataset (DataLoader or str): dataset variable if mode='save',
                                         name of the file if mode='load'
            mode (str): 'save' or 'load', the dataset is saved or loaded
                        (default = 'save', .pt file, filepath defined inside)
            dataset_type (str): 'train' for train dataset, 'valid' for
                                 validation dataset or 'test' fot test
                                 dataset(default = 'train')
            
        Return (if mode='load'):
            dataset (DataLoader): dataset
        """
        
        filepath = "D:/Home/File/Python_Project/App_GUI_DL/Model_n_Datasets/"
        
        if mode == 'save':
            torch.save(dataset, filepath + dataset_name)
            print("Dataset saved...")
        elif mode == 'load':
            dat = torch.load(filepath + dataset_name)
            print("Dataset loaded...")
            return dat
        else:
            raise ValueError("mode must be equal to 'save' or 'load'.")


# end