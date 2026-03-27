import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !pip install torchmetrics
import torchmetrics


def get_train_valid_loader(data_dir, batch_size, augment):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip()
            if augment
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            normalize,
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Load separate datasets for train and valid
    train_dataset = datasets.CelebA(
        root=data_dir, split="train", download=False, transform=train_transform
    )
    valid_dataset = datasets.CelebA(
        root=data_dir, split="valid", download=False, transform=valid_transform
    )

    # Use standard DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, valid_loader


def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    # define transform
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = datasets.CelebA(
        root=data_dir,
        split="test",
        download=False,
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


train_loader, valid_loader = get_train_valid_loader(
    data_dir="dataset/", batch_size=64, augment=False
)

test_loader = get_test_loader(data_dir="dataset/", batch_size=64)

# Re-calculate total_step after the loader is actually created
total_step = len(train_loader)


class lee(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(lee, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


num_classes = 40
num_epochs = 20
batch_size = 64
learning_rate = 0.001

model = lee(3, num_classes).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9
)

# Train the model
total_step = len(train_loader)

total_step = len(train_loader)


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device).float()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add this every 100 batches:
        #        if (i + 1) % 100 == 0:
        #   print(
        #       f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}"
        #   )

        #    print(
        # "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
        #    epoch + 1, num_epochs, i + 1, total_step, loss.item()
        # )
    # )
    # Commented Out for ease of comparison

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

        accuracy = 100 * correct / total
        print(
            "Accuracy of the network on the {} validation images: {} %".format(
                5000, accuracy
            )
        )





def svd_approx(kernel, rank):  # Approximates to get the important data and trim 0s
    original_shape = kernel.shape

    k2d = kernel.reshape(original_shape[0], -1)

    # U: Output Spcae
    # S: Singular Values
    # Vh: Input Space
    U, S, Vh = torch.linalg.svd(k2d, full_matrices=False)

    # Chooses WHICH index of U,S, and Vh
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    k2d_approx = U_r @ torch.diag(S_r) @ Vh_r
    return k2d_approx.reshape(original_shape)


ranks1 = [round(r) for r in np.linspace(1, 6, 20)]  # max is min(6, 75) = 6 for layer1
ranks2 = [
    round(r) for r in np.linspace(1, 16, 20)
]  # max is min(16, 150) = 16 for layer2
# Chose the Max for best output of the Accuracy

for r1, r2 in zip(ranks1, ranks2):
    with torch.no_grad():
        for layer in model.layer1:
            if hasattr(layer, 'weight') and layer.weight is not None:
                kernel = layer.weight.data.clone()
                layer.weight.copy_(svd_approx(kernel, r1))

        for layer in model.layer2:
            if hasattr(layer, 'weight') and layer.weight is not None:
                kernel = layer.weight.data.clone()
                layer.weight.copy_(svd_approx(kernel, r2))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

        svd_accuracy = 100 * correct / total
        print(
            "Accuracy of the network on the {} validation images (SVD rank {}/{}): {} %".format(
                5000, r1, r2, svd_accuracy
            )
        )
