#Packages we need for the project 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, stat
import json
import argparse
from collections import OrderedDict
from PIL import Image
# train import Argument_Parser

import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models

# Initiate variables with default values
checkpoint = 'checkpoint.pth'
filepath = 'cat_to_name.json'    
arch=''
image_path = 'flowers/test/100/image_07896.jpg'
topk = 5

# Set up parameters for entry in command line
parser = argparse.ArgumentParser()
parser.add_argument('-c','--Checkpoint', action='store',type=str, default="/checkpoint.pth", help=' model to be loaded and used for predictions in your working directory.')
parser.add_argument('-i','--image_path',action='store',type=str, help='Location of image to predict ')
parser.add_argument('-k', '--topk', action='store',type=int, help='Numbers of classes.')
parser.add_argument('-j', '--json', action='store',type=str, default='cat_to_name.json', help=' file in json format that holding class names.')
parser.add_argument('-g','--gpu', action='store', dest="gpu", help='Use GPU if available')

args = parser.parse_args()

# Select parameters entered in command line
if args.checkpoint:
    checkpoint = args.checkpoint
if args.image_path:
    image_path = args.image_path
if args.topk:
    topk = args.topk
if args.json:
    filepath = args.json
if args.gpu:
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(filepath, 'r') as f:
    cat_to_name = json.load(f)

def load_model(checkpoint_path):

    checkpoint_path = os.getcwd()+Checkpoint
    os.chmod(checkpoint_path, stat.S_IRWXU)
 
    checkpoint = torch.load(save_dir)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 25088
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = 9216
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = 1024
        for param in model.parameters():
            param.requires_grad = False
    else:
        print('This model architecture does not exist in this catalogue')
    
    model.class_to_idx = checkpoint['class_to_idx']
    hidden_units = checkpoint['hidden_units']
    
    classifier = nn.Sequential(OrderedDict([
                           ('fc1',nn.Linear(in_features,hidden_units)),
                           ('ReLu1',nn.ReLU()),
                           ('Dropout1',nn.Dropout(p=0.5)),
                           ('fc2',nn.Linear(hidden_units,512)),
                           ('ReLu2',nn.ReLU()),
                           ('Dropout2',nn.Dropout(p=0.5)),
                           ('fc3',nn.Linear(512,102)),
                           ('output',nn.LogSoftmax(dim=1))
                           ]))    
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image_path):

    
    # Process a PIL image for use in a PyTorch model
    size = 256, 256
    crop_size = 224
    
    image = Image.open(image_path)
    
    image.thumbnail(size)

    left = (size[0] - crop_size)/2
    top = (size[1] - crop_size)/2
    right = (left + crop_size)
    bottom = (top + crop_size)

    im = image.crop((left, top, right, bottom))
    
    np_image = np.array(im)
    np_image = np_image/255
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    np_image = (np_image - mean) / std
    pytorch_np_image = np_image.transpose(2,0,1)
    
    return pytorch_np_image

def predict(image_path, model, topk=5):
   
    
    # Use process_image function to create numpy image tensor
    pytorch_np_image = process_image(image_path)
    
    # Changing from numpy to pytorch tensor
    pytorch_tensor = torch.tensor(pytorch_np_image)
    pytorch_tensor = pytorch_tensor.float()
    
    # Removing RunTimeError for missing batch size - add batch size of 1 
    pytorch_tensor = pytorch_tensor.unsqueeze(0)
    
    # Run model in evaluation mode to make predictions
    model.eval()
    LogSoftmax_predictions = model.forward(pytorch_tensor)
    predictions = torch.exp(LogSoftmax_predictions)
    
    # Identify top predictions and top labels
    top_preds, top_labs = predictions.topk(topk)
    
    
    top_preds = top_preds.detach().numpy().tolist()
    
    top_labs = top_labs.tolist()
    
    labels = pd.DataFrame({'class':pd.Series(model.class_to_idx),'flower_name':pd.Series(cat_to_name)})
    labels = labels.set_index('class')
    labels = labels.iloc[top_labs[0]]
    labels['predictions'] = top_preds[0]
    
    return labels

model = load_model(checkpoint) 


print(model)
labels = predict(image_path,model,topk)
print(labels)
