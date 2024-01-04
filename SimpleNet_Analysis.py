# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 01:30:07 2023

@author: Edoardo Giancarli
"""

import os
os.chdir("D:/Home/File/Python_Project/App_GUI_DL/")    # path with modules

import torch         
from torchvision import transforms
import pysim as psm

# model analysis
#
# model initialisation on a dataset subset


# warm up gpu
# psm._WarmUpGpu().warm_up()


# define model
model = psm.SimpleNet(act_func='ReLU')
model_id = '_'+'model2'+'_'+'test2'+'_'+'trial10'

# call SimpleTools
smt = psm.SimpleTools()

# model output classes
class_names = {0: 'triangle',
               1: 'square',
               2: 'circle'}

# transform images
transform = transforms.Compose([transforms.functional.to_grayscale,
                                transforms.ToTensor(),
                                transforms.Resize((36, 36), antialias=True),
                                psm.Normalize()])

# datasets
train_path = "D:/Home/File/Python_Project/App_GUI_DL/Images/Img_Train/"
test_path = "D:/Home/File/Python_Project/App_GUI_DL/Images/Img_Test/"
batch_size = 250

try:
    train_dataset = smt.manage_dataset(dataset_name='train_dataset.pt', mode='load') # ' + model_id + '
    valid_dataset = smt.manage_dataset(dataset_name='valid_dataset.pt', mode='load')
    test_dataset = smt.manage_dataset(dataset_name='test_dataset.pt', mode='load')
except:
    train_dataset, valid_dataset = smt.make_dataset(train_path, batch_size, class_names, valid_size=0.4, transform=transform)
    test_dataset = smt.make_dataset(test_path, batch_size, class_names, transform=transform)


# train model
torch.cuda.empty_cache()

epochs = [5, 7, 10, 10]
lr = [5e-3, 1e-3, 1e-4, 1e-5]

model, tl1, ta1, vl1, va1 = smt.train_model(model, epochs=epochs[0], learn_rate=lr[0],
                                            train_dataset=train_dataset, valid_dataset=valid_dataset)
# smt.show_model()

model, tl2, ta2, vl2, va2 = smt.train_model(model, epochs=epochs[1], learn_rate=lr[1],
                                            train_dataset=train_dataset, valid_dataset=valid_dataset)
# smt.show_model()

model, tl3, ta3, vl3, va3 = smt.train_model(model, epochs=epochs[2], learn_rate=lr[2],
                                            train_dataset=train_dataset, valid_dataset=valid_dataset)
# smt.show_model()

model, tl4, ta4, vl4, va4 = smt.train_model(model, epochs=epochs[3], learn_rate=lr[3],
                                            train_dataset=train_dataset, valid_dataset=valid_dataset)
# smt.show_model()


# plot model loss and accuracy
# print(model)
train_loss = tl1 + tl2 + tl3 + tl4 #+ tl5 #+ tl6
train_accuracy = ta1 + ta2 + ta3 + ta4 #+ ta5 #+ ta6

valid_loss = vl1 + vl2 + vl3 + vl4 #+ vl5 #+ vl6
valid_accuracy = va1 + va2 + va3 + va4 #+ va5 #+ va6

# loss and accuracy plot
smt.show_model(train_loss=train_loss, train_accuracy=train_accuracy, valid_loss=valid_loss,
               valid_accuracy=valid_accuracy, title_notes=None, comp_dloss=False)


# test model
_ = smt.test_model(test_dataset)


# save model
filename = 'SimpleNet' + model_id + '.pth'
notes = "On the right road."
gpu = '1206MiB' + ' / 6144MiB'

smt.save_model(batch_size, epochs, lr, activation='ReLU', filename=filename,
               train_loss=train_loss, train_accuracy=train_accuracy, valid_loss=valid_loss,
               valid_accuracy=valid_accuracy, notes=notes, gpu=gpu)


# save datasets
smt.manage_dataset(train_dataset, 'train_dataset.pt', mode='save') # ' + model_id + '
smt.manage_dataset(valid_dataset, 'valid_dataset.pt', mode='save')
smt.manage_dataset(test_dataset, 'test_dataset.pt', mode='save')


# load model
smt = psm.SimpleTools()
model2, obs = smt.load_model('SimpleNet_model2_test2_trial1.pth')

###############################################################################


# train model multiple times and average it
smt = psm.SimpleTools()

model = psm.SimpleNet(act_func='ReLU')
n = 5
epochs = [5, 7, 10, 10]
lr = [5e-3, 1e-3, 1e-4, 1e-5]
train_dataset = smt.manage_dataset(dataset_name='train_dataset.pt', mode='load')
test_dataset = smt.manage_dataset(dataset_name='test_dataset.pt', mode='load')
batch_size = 250
activation = 'ReLU'
model_ID = 'frufru'

psm.train_multiple_models(model, n, epochs, lr, train_dataset, test_dataset, batch_size, activation, model_ID)

psm.MeanModel()

# probs histogram
images_path = "D:/Home/File/Python_Project/App_GUI_DL/Images/Img_Test/"

class_names = {0: 'triangle',
               1: 'square',
               2: 'circle'}

transform = transforms.Compose([transforms.functional.to_grayscale,
                                transforms.ToTensor(),
                                transforms.Resize((36, 36), antialias=True),
                                psm.Normalize()])

psm.StatTools().probs_histogram(images_path, model2, class_names, transform)



###############################################################################
## model initialisation on a dataset subset

import os
os.chdir("D:/Home/File/Python_Project/App_GUI_DL/")    # path with modules

import torch         
from torchvision import transforms
import pysim as psm

# warm up gpu
# psm._WarmUpGpu().warm_up()


# define model
model = psm.SimpleNet(act_func='PReLU')

# call SimpleTools
smt = psm.SimpleTools()

# model output classes
classes_names = {0: 'triangle',
                 1: 'square',
                 2: 'circle'}

# transform images
transform = transforms.Compose([transforms.functional.to_grayscale,
                                transforms.ToTensor(),
                                transforms.Resize((36, 36), antialias=True),
                                psm.Normalize()])

# datasets
train_path = "D:/Home/File/Python_Project/App_GUI_DL/Images/Img_Train/"
train_dataset, valid_dataset = smt.make_dataset(train_path, batch_size, classes_names, valid_size=0.99, transform=transform)

# train model on subset
torch.cuda.empty_cache()

_ = smt.train_model(model, epochs=50, learn_rate=1e-3, train_dataset=train_dataset, valid_dataset=valid_dataset)
smt.show_model()

del model

# end