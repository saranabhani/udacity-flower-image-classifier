import json
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils import data
import numpy as np
from PIL import Image

def data_transforms(data_set):
    if data_set == 'train':
        return transforms.Compose([transforms.RandomRotation(35),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
    else:
        return transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

def image_folder(data_set, data_dir):
    return ImageFolder(root=data_dir, transform=data_transforms(data_set))

def data_loaders(data_set, data_dir):
    if data_set == 'train':
        return data.DataLoader(image_folder(data_set, data_dir), batch_size=32, shuffle=True)
    else:
        return data.DataLoader(image_folder(data_set, data_dir), batch_size=32)
    
def cat_2_name(file):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = models.vgg16(pretrained=True)

    model.classifier = nn.Sequential(nn.Linear(25088, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1024, 102),
                        nn.LogSoftmax(dim=1))

    model.load_state_dict(checkpoint['state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']


    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Open the image
    img = Image.open(image)

    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # Normalize
    np_image = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))

    return np_image