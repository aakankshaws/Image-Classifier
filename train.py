# Imports libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models,transforms
from collections import OrderedDict
from PIL import Image
import json
import argparse

parser = argparse.ArgumentParser(description='train-file')

parser.add_argument('data_dir', type = str, action="store", default="./flowers/")
parser.add_argument('--category_names ', type = str, action="store",  dest="category_names", default="./cat_to_name.json")
parser.add_argument('--arch', type = str, action="store", dest="arch", default='vgg16')
parser.add_argument('--learning_rate',  action="store", dest="lr", default=0.03)
parser.add_argument('--hidden_units',  action="store", dest="hidden_units", default=500)
parser.add_argument('--epochs',  action="store", dest="epochs", default=4)
parser.add_argument('--gpu',  action="store", dest="gpu", default='gpu')
parser.add_argument('--save_dir ',  action="store", dest="save_dir", default='./checkpoint.pth')

results = parser.parse_args()

data_dir = results.data_dir
category_names = results.category_names
arch = results.arch
lr = results.lr
hidden_units = results.hidden_units 
epochs = results.epochs
gpu = results.gpu
save_dir = results.save_dir

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, validation_transforms)
test_datasets = datasets.ImageFolder(test_dir, test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True) 
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True) 
testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True) 

def label_map():
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

# Use GPU if it's available
device = torch.device("cuda" if (torch.cuda.is_available() and gpu=='gpu') else "cpu")
print("We are running on {}".format(device)) #print to see what device is running

def build_train(arch):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet121':
         model = models.densenet121(pretrained=True)
         input_size = model.classifier.in_features  
            
    #freeze the model parameters by setting requires_grad to False, so that we don't backprop through them.
    for param in model.parameters():
        param.requires_grad = False
        
    #classifier input size and output_size
    output_size = len(label_map()) #no. of classes 

    #Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of the network.
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(hidden_units, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier 
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device);
    return model, criterion, optimizer

model, criterion, optimizer = build_train(arch)

def do_deep_learning(model, trainloader, testloader, epochs, print_every, criterion, optimizer, device):
    epochs = epochs
    print_every = print_every
    steps = 0
    running_loss = 0

    train_losses, test_losses = [], []
    for e in range(epochs):
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss = 0
                    accuracy = 0
                    for images, labels in testloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        test_loss += criterion(logps, labels).item()

                        #calculate accuracy
                        ps = torch.exp(logps)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()

                        
                print("Training Network")
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                
                # Make sure training is back on
                model.train()
     
do_deep_learning(model, trainloaders, validloaders, epochs, 40, criterion, optimizer, device)    
print("We have finished training the network")

# TODO: Do validation on the test set
def network_accuracy(testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()            
            
    print('Accuracy of the network: %d %%' % (100 * correct / total))
    
network_accuracy(testloaders, device)      

# TODO: Save the checkpoint
model.class_to_idx = train_datasets.class_to_idx
model.cpu()
torch.save({'arch': arch,
            'hidden_units': hidden_units,
            'lr': lr,
            'epochs': epochs,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            save_dir)

print("checkpoint saved in {}".format(save_dir))