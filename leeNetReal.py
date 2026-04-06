import matplotlib.pyplot as plt
import copy
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.amp import autocast, GradScaler
import os
import time

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "./data_faces"

# ── Transforms ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
VGG2Dataset = datasets.ImageFolder(output_dir, transform=transform)
num_classes  = len(VGG2Dataset.classes)

torch.manual_seed(42)
train_size = int(len(VGG2Dataset) * 0.75)
test_size  = len(VGG2Dataset) - train_size
train_set, test_set = torch.utils.data.random_split(VGG2Dataset, [train_size, test_size])

test_set.dataset           = copy.deepcopy(VGG2Dataset)
test_set.dataset.transform = test_transform

# ── Model ─────────────────────────────────────────────────────────────────────
class LeNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

# ── SVD Layer Factorization ───────────────────────────────────────────────────
def make_svd_linear(layer, rank_ratio=0.5):
    W = layer.weight.data
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    rank = max(1, round(S.numel() * rank_ratio))
    print(f"    Linear ({W.shape[0]}x{W.shape[1]}) → rank {rank} | "
          f"params: {W.numel():,} → {rank*(W.shape[0]+W.shape[1]):,}")
    S_sqrt = torch.diag(torch.sqrt(S[:rank]))
    A = U[:, :rank] @ S_sqrt
    B = S_sqrt @ Vh[:rank, :]
    layer1 = nn.Linear(W.shape[1], rank, bias=False)
    layer2 = nn.Linear(rank, W.shape[0], bias=layer.bias is not None)
    layer1.weight.data = B
    layer2.weight.data = A
    if layer.bias is not None:
        layer2.bias.data = layer.bias.data.clone()
    return nn.Sequential(layer1, layer2)

def make_svd_conv(layer, rank_ratio=0.5):
    W = layer.weight.data
    out_c, in_c, kH, kW = W.shape
    W2d = W.reshape(out_c, -1)
    U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)
    rank = max(1, round(S.numel() * rank_ratio))
    print(f"    Conv2d ({out_c}x{in_c}x{kH}x{kW}) → rank {rank} | "
          f"params: {W.numel():,} → {rank*(out_c + in_c*kH*kW):,}")
    S_sqrt = torch.diag(torch.sqrt(S[:rank]))
    A = U[:, :rank] @ S_sqrt
    B = S_sqrt @ Vh[:rank, :]
    conv1 = nn.Conv2d(in_c, rank, kernel_size=(kH, kW),
                      stride=layer.stride, padding=layer.padding, bias=False)
    conv2 = nn.Conv2d(rank, out_c, kernel_size=1, bias=layer.bias is not None)
    conv1.weight.data = B.reshape(rank, in_c, kH, kW)
    conv2.weight.data = A.reshape(out_c, rank, 1, 1)
    if layer.bias is not None:
        conv2.bias.data = layer.bias.data.clone()
    return nn.Sequential(conv1, conv2)

def replace_layers_with_svd(model, rank_ratio=0.5):
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Sequential):
            new_layers = []
            for i, layer in enumerate(module):
                if isinstance(layer, nn.Linear):
                    print(f"  {name}[{i}] Linear:")
                    new_layers.append(make_svd_linear(layer, rank_ratio))
                elif isinstance(layer, nn.Conv2d):
                    print(f"  {name}[{i}] Conv2d:")
                    new_layers.append(make_svd_conv(layer, rank_ratio))
                else:
                    new_layers.append(layer)
            setattr(model, name, nn.Sequential(*new_layers))
        elif isinstance(module, nn.Linear):
            print(f"  {name}:")
            setattr(model, name, make_svd_linear(module, rank_ratio))
        elif isinstance(module, nn.Conv2d):
            print(f"  {name}:")
            setattr(model, name, make_svd_conv(module, rank_ratio))
        else:
            replace_layers_with_svd(module, rank_ratio)

# ── Helpers ───────────────────────────────────────────────────────────────────
def evaluate(model, loader, label="Test"):
    correct_top1, correct_top5, total = 0, 0, 0
    top_k = min(5, num_classes)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred_top1 = torch.max(outputs, 1)
            correct_top1 += (pred_top1 == labels).sum().item()
            _, pred_top5 = outputs.topk(top_k, dim=1)
            correct_top5 += sum(labels[i] in pred_top5[i] for i in range(len(labels)))
            total += labels.size(0)
    print(f"{label} | Top-1: {100*correct_top1/total:.2f}% | Top-5: {100*correct_top5/total:.2f}%")

def model_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def benchmark(model, loader, label="Model", warmup_batches=10):
    model.eval()
    top_k = min(5, num_classes)

    print(f"Warming up {label}...")
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            _ = model(images.to(device))
            torch.cuda.synchronize()
            if i >= warmup_batches:
                break

    correct_top1, correct_top5, total = 0, 0, 0
    batch_times = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(images)
            torch.cuda.synchronize()
            batch_times.append(time.perf_counter() - t0)
            _, pred_top1 = torch.max(outputs, 1)
            correct_top1 += (pred_top1 == labels).sum().item()
            _, pred_top5 = outputs.topk(top_k, dim=1)
            correct_top5 += sum(labels[i] in pred_top5[i] for i in range(len(labels)))
            total += labels.size(0)

    top1       = 100 * correct_top1 / total
    top5       = 100 * correct_top5 / total
    avg_ms     = (sum(batch_times) / len(batch_times)) * 1000
    p95_ms     = sorted(batch_times)[int(len(batch_times) * 0.95)] * 1000
    throughput = total / sum(batch_times)
    params     = count_params(model)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Top-1 Accuracy    : {top1:.2f}%")
    print(f"  Top-5 Accuracy    : {top5:.2f}%")
    print(f"  Avg batch latency : {avg_ms:.2f} ms")
    print(f"  p95 batch latency : {p95_ms:.2f} ms")
    print(f"  Throughput        : {throughput:.1f} images/sec")
    print(f"  Parameters        : {params:,}")
    print(f"{'='*55}\n")

    return {
        "label": label, "top1": top1, "top5": top5,
        "avg_ms": avg_ms, "p95_ms": p95_ms,
        "throughput": throughput, "params": params
    }

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Classes (identities): {num_classes}")
    print(f"Total images: {len(VGG2Dataset)}")

    # ── DataLoaders (inside guard for Windows) ────────────────────────────────
    trainLoader = torch.utils.data.DataLoader(
        train_set, batch_size=256, shuffle=True,
        pin_memory=True, num_workers=4,
        persistent_workers=True
    )
    testLoader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False,
        pin_memory=True, num_workers=4,
        persistent_workers=True
    )
    print(f"Train: {len(train_set)} | Test: {len(test_set)}")

    # ── CUDA warmup ───────────────────────────────────────────────────────────
    model = LeNet(num_classes=num_classes).to(device)
    print(f"num_classes: {num_classes} | output shape: {list(model.parameters())[-1].shape}")
    print(f"Parameters: {count_params(model):,}")

    print("Warming up CUDA...")
    dummy = torch.zeros(1, 3, 32, 32).to(device)
    for _ in range(3):
        _ = model(dummy)
    torch.cuda.synchronize()
    print("CUDA ready.\n")

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer  = optim.Adam(model.parameters(), lr=0.0004)
    criterion  = nn.CrossEntropyLoss()
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    scaler     = GradScaler('cuda')
    epochs     = 50
    train_loss = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        t0 = time.time()

        for idx, (image, label) in enumerate(trainLoader):
            if idx == 0 and epoch == 0:
                print(f"  First batch received after {time.time()-t0:.1f}s")
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                loss = criterion(model(image), label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

        scheduler.step()
        avg_loss = total_train_loss / (idx + 1)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch:>2} | Loss: {avg_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {time.time()-t0:.1f}s")

    plt.plot(train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss - LeNet")
    plt.savefig("train_loss_lenet.png")

    # ── Save original trained model ───────────────────────────────────────────
    torch.save(model.state_dict(), "lenet_trained.pth")
    print(f"\nSaved → lenet_trained.pth ({model_size_mb('lenet_trained.pth'):.1f} MB)")
    print(f"Original params: {count_params(model):,}")

    evaluate(model, testLoader, "Test — Before SVD")

    # ── SVD factorization ─────────────────────────────────────────────────────
    print("\n=== SVD Layer Replacement (rank_ratio=0.5) ===")
    replace_layers_with_svd(model, rank_ratio=0.5)
    model = model.to(device)

    torch.save(model.state_dict(), "lenet_svd.pth")
    print(f"\nSaved → lenet_svd.pth ({model_size_mb('lenet_svd.pth'):.1f} MB)")
    print(f"SVD params: {count_params(model):,}")

    evaluate(model, testLoader, "Test — After SVD")

    # ── Benchmark all three ───────────────────────────────────────────────────
    model_original = LeNet(num_classes=num_classes).to(device)
    model_original.load_state_dict(torch.load("lenet_trained.pth", weights_only=True))

    model_svd = LeNet(num_classes=num_classes)
    replace_layers_with_svd(model_svd, rank_ratio=0.5)
    model_svd.load_state_dict(torch.load("lenet_svd.pth", weights_only=True))
    model_svd = model_svd.to(device)

    results = []
    results.append(benchmark(model_original, testLoader, "LeNet original"))
    results.append(benchmark(model_svd,      testLoader, "LeNet SVD"))

    baseline = results[0]
    print(f"\n{'Model':<20} {'Top-1':>7} {'Top-5':>7} {'ms/batch':>10} {'img/sec':>10} {'params':>12} {'speedup':>9}")
    print("-" * 80)
    for r in results:
        speedup = baseline['avg_ms'] / r['avg_ms']
        print(f"{r['label']:<20} {r['top1']:>6.2f}% {r['top5']:>6.2f}% "
              f"{r['avg_ms']:>9.2f} {r['throughput']:>9.1f} "
              f"{r['params']:>12,} {speedup:>8.2f}x")