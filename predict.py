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

parser = argparse.ArgumentParser(description='predict-file')

parser.add_argument('input', type = str, action="store", default="flowers/test/99/image_07833.jpg")
parser.add_argument('checkpoint', type = str, action="store", default="./checkpoint.pth")
parser.add_argument('--top_k', dest='top_k', type = str, action="store", default=5)

results = parser.parse_args()

image_path = results.input
checkpoint = results.checkpoint
topk = results.top_k

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet121':
         model = models.densenet121(pretrained=True)
         input_size = model.classifier.in_features 
    
    output_size = len(cat_to_name)
    hidden_units = checkpoint['hidden_units']
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(hidden_units, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model = load_checkpoint(checkpoint)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img_adjust = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    return img_adjust(img)

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image_tensor = image.type(torch.FloatTensor)
    model_input = image_tensor.unsqueeze(0)
    probs = torch.exp(model.forward(model_input))
    
    top_probs, top_labs = probs.topk(5)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers

top_probs, top_labels, top_flowers = predict(image_path, model, topk)

i=0
while i < topk:
    print("{} with a probability of {}".format(top_flowers[i], top_probs[i]))
    i += 1

