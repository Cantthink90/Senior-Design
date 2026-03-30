import matplotlib.pyplot as plt
import numpy as np
import copy
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from collections import Counter

import zipfile

import shutil
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import pandas as pd

device = ("cuda" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

output_dir = "../data_faces"


transform = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.RandomHorizontalFlip(),
          transforms.RandomRotation(10),
          transforms.ColorJitter(brightness=0.2, contrast=0.2),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
          ])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

VGG2Dataset = datasets.ImageFolder(output_dir, transform=transform)
num_classes = len(VGG2Dataset.classes)
print(f"Classes (identities): {num_classes}")
print(f"Total images: {len(VGG2Dataset)}")

torch.manual_seed(42)
train_size = int(len(VGG2Dataset) * 0.75)
test_size = len(VGG2Dataset) - train_size
train_set, test_set = torch.utils.data.random_split(VGG2Dataset, [train_size, test_size])

test_set.dataset = copy.deepcopy(VGG2Dataset)
test_set.dataset.transform = test_transform

trainLoader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True)
testLoader  = torch.utils.data.DataLoader(test_set,  batch_size=64, shuffle=False, pin_memory=True)

print(f"Train: {len(train_set)} | Test: {len(test_set)}")

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 2):
        super(AlexNet, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.convolutional(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

model = AlexNet(num_classes = num_classes).to(device)
print(f"num_classes: {num_classes} | model output: {list(model.parameters())[-1].shape}")

#training
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
epochs = 50
train_loss = []


for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for idx, (image, label) in enumerate(trainLoader):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, label)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    scheduler.step()
    avg_loss = total_train_loss / (idx + 1)
    train_loss.append(avg_loss)
    print(f'Epoch: {epoch} | Train Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

plt.plot(train_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss - Untrained AlexNet")
plt.savefig("train_loss_untrained.png")


#Evaluation
def evaluate(loader, label="Test"):
    correct_top1, correct_top5, total = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Top-1
            _, pred_top1 = torch.max(outputs, 1)
            correct_top1 += (pred_top1 == labels).sum().item()

            # Top-5
            _, pred_top5 = outputs.topk(5, dim=1)
            correct_top5 += sum(labels[i] in pred_top5[i] for i in range(len(labels)))

            total += labels.size(0)

    print(f"{label} | Top-1: {100*correct_top1/total:.2f}% | Top-5: {100*correct_top5/total:.2f}%")

evaluate(testLoader, "Test Before SVD")


def svd_approx(kernel):
    original_shape = kernel.shape
    k2d = kernel.reshape(original_shape[0], -1)
    U, S, Vh = torch.linalg.svd(k2d, full_matrices=False)
    rank = round(S.numel() * 0.5)
    print(f"full rank: {S.numel()} | kept rank: {rank}")
    k2d_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
    return k2d_approx.reshape(original_shape)

with torch.no_grad():
    for layer in list(model.convolutional) + list(model.linear):
        if hasattr(layer, 'weight') and layer.weight is not None:
            layer.weight.copy_(svd_approx(layer.weight.data.clone()))

evaluate(testLoader, "Test After SVD")