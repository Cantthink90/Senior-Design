from leeTestSVDAllKern import lee, test_loader

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !pip install torchmetrics

PATH = "lee_trained.pth"


# Assume 'TheModelClass' is defined with the correct architecture
model = lee(3, 40)
model.load_state_dict(torch.load(PATH))
model.to(device)
model.eval()  # Set to evaluation mode for inference


def svd_approx(kernel):  # Approximates to get the important data and trim 0s
    original_shape = kernel.shape

    k2d = kernel.reshape(original_shape[0], -1)

    # U: Output Spcae
    # S: Singular Values
    # Vh: Input Space
    U, S, Vh = torch.linalg.svd(k2d, full_matrices=False)
    rank = round((S.numel()) * 0.3)
    print(rank)
    # Chooses WHICH index of U,S, and Vh
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    k2d_approx = U_r @ torch.diag(S_r) @ Vh_r
    return k2d_approx.reshape(original_shape)


for layer in model.layer1:
    if hasattr(layer, "weight") and layer.weight is not None:
        kernel = layer.weight.data.clone()
        layer.weight.data.copy_(svd_approx(kernel))

for layer in model.layer2:
    if hasattr(layer, "weight") and layer.weight is not None:
        kernel = layer.weight.data.clone()
        layer.weight.data.copy_(svd_approx(kernel))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).float()
        outputs = model(images)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.numel()

    svd_accuracy = 100 * correct / total
    print(
        "Accuracy of the SVD network on the {} validation images : {} %".format(
            5000, svd_accuracy
        )
    )
