import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# !pip install torchvision
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import v2

# !pip install torchmetrics
import torchmetrics

batch_size = 60

transform = v2.Compose([
    v2.Resize((224, 224)),  # Example: resize images to a fixed size
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

train_dataset = datasets.CelebA(root="dataset/", download=True, split='train', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CelebA(root="dataset/", download=True, split='valid', transform=transform)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# get some random training images
image, targets = random.choice(test_dataset)
# show images
plt.imshow(np.transpose(image,(1,2,0)))
# plt.title(f"Image from WIDERFace Dataset")
# plt.axis('off') # Hide axis values
plt.show()

class lee(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(lee, self).__init__()

class alex(nn.Module):
    def __intit__(self, in_channels, num_classes):
        super(alex, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU())
        self.layer4 = nn.Sequential(
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU())
        self.layer5 = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(9216, 4096),
                nn.ReLU())
        self.fc1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU())
        self.fc2= nn.Sequential(
                nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out