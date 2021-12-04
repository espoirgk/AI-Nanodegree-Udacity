#Packages we need for the project 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from collections import OrderedDict
from PIL import Image

import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models


#Use ArgumentParser function that will hold all the information necessary for
# entry in command line
def Argument_Parser():
    """ Set up parameters we need to the parser"""
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--data_dir', type=str, default='flowers', help='Path of directory with data for the image')
    parser.add_argument('--save_dir',  dest='save_dir', action='store', type=str, default="/checkpoint.pth", help='filename where to save the trained model')
    parser.add_argument('--arch', dest='arch', action='store', type=str, default='vgg16', help='Choose between the 3 pretrained networks model: vgg16, alexnet, densenet121')
    parser.add_argument('--hidden_units', dest='hidden_units', action='store',type=int, default=120, help='Give a number of hidden units for the first layer')
    parser.add_argument('--learning_rate', dest='learning_rate', action='store',type=float, default=0.001, help='Give a float number for the learning rate')
    parser.add_argument('--epochs', dest='epochs', action='store',type=int, default=1, help='Give a number of epochs')
    parser.add_argument('--gpu', dest='gpu', action='store', default="gpu", help='Use GPU if available')
    args = parser.parse_args()
    return args

# Set all path for location of images
args = Argument_Parser()
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#

#Define 3 functions that contains ordered dictionnary for the transformers for the training, validation, and testing sets


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
}

# Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform = data_transforms["train"]),
    'valid': datasets.ImageFolder(valid_dir, transform = data_transforms["valid"]),
    'test': datasets.ImageFolder(test_dir, transform = data_transforms["test"])
}


# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=48, shuffle=True)
              for x in ['train', 'valid', 'test']}
#Check for dataset sizes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

#Define gpu or cpu for the job
if args.gpu:        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load the model with this function

def load_model(arch='vgg16', hidden_units=25088,learning_rate=0.001):
    print("----> Your model and its parameters are loading...<----")
    
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"

    #Here we freeze params of vg66 model
    for param in model.parameters():
        param.requires_grad = False
    

    #Create classifier by putting it in orderedDict
    classifier = nn.Sequential(OrderedDict([
            ('inputs', nn.Linear(25088, 240)), 
            ('relu1', nn.ReLU()),
            ('dropout',nn.Dropout(0.5)),
            ('hidden_layer1', nn.Linear(240, 120)), 
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(120,80)), 
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    #scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=6,gamma=0.1)
    print("---->  loading OK...<----")
    return model, optimizer, criterion

model, optimizer, criterion = load_model(args.arch, args.hidden_units, args.learning_rate)


#Train and validate the model with this function

def train_model(model, optimizer, criterion, n_epochs):
    print("----> Training starts...<----")
    model.to(device)
    

    # Iteration on each epochs for training and validation
    n_epochs=args.epochs
    n_steps = 0
    for epoch in range(n_epochs):
        print("Epoch: {}/{}".format(epoch+1, n_epochs))
        loss_train = 0.0
        correct_train = 0
        n_steps += 1
        model.train(True)

        for inputs, labels in dataloaders["train"]:
            
            inputs = inputs.to(device)
            labels = labels.to(device)

             # Reset every gradien on zero
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            
            
            if n_steps % n_epochs == 0:
                model.eval()
                with torch.no_grad():
                            valid_loss = 0
                            valid_accuracy = 0
                            
                            for inputs, labels in dataloaders["valid"]:
                                inputs = inputs.to(device)
                                labels = labels.to(device)
                                                           
                                output = model.forward(inputs)
                                valid_loss += criterion(output, labels).item()
                                
                                ps = torch.exp(output)
                                equality = (labels.data == ps.max(dim=1)[1])
                                valid_accuracy += equality.type(torch.FloatTensor).mean()
                                
                                
                print("Training Loss: {:.4f} == ".format(loss_train/n_steps),
                     "Validation Loss: {:.4f} == ".format(valid_loss/dataset_sizes["valid"]),
                     "Validation Accuracy: {:.4f}".format(valid_accuracy/dataset_sizes["valid"]))
                
                loss_train = 0
                model.train()

    print("----> Training end...<----")            
    
    return model
    


#function that loads a checkpoint and rebuilds the model

best_model_trained = train_model(model, optimizer, criterion, n_epochs=args.epochs)

best_model_trained


#Testing the model on test data
def evaluation(model):
    correct_prediction = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in dataloaders["test"]:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            _, predictions = torch.max(outputs.data,1)
            total += labels.size(0)
            correct_prediction += (predictions == labels).sum().item()
            accuracy = 100 * correct_prediction / total
        print('The accuracy of test is: {:.2f}%'.format(accuracy))

evaluation(best_model_trained)

#function to load the checkpoint


def init_checkpoint(best_model_trained):
    import os
    print("----> Checkpoint initialization starts...<----")
    best_model_trained.class_to_idx = image_datasets['train'].class_to_idx
    
    model.cpu()
    checkpoint = {'arch' :'densenet121',
            'epochs':15,
            'hidden_layer1':args.hidden_units,
            'optimizer_dict': optimizer.state_dict(),
            'state_dict':best_model_trained.state_dict(),
            'class_to_idx':best_model_trained.class_to_idx}
    import os, sys, stat
    
    save_dir = os.getcwd()+args.save_dir
    
    os.chmod(save_dir, stat.S_IRWXU)
    print("Your checkpoint is here: {0}".format(os.getcwd()))

    
    torch.save(checkpoint, save_dir)

init_checkpoint(best_model_trained)
print("----> Model saved successfully...<----")