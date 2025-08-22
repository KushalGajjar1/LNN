import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import argparse
import random
import numpy as np
import time

try:
    from thop import profile as thop_profile
    THOP_AVAILABLE = True
except Exception:
    THOP_AVAILABLE = False

class LiquidNeuron(nn.Module):

    def __init__(self, input_size, hidden_size, dt=0.1):
        super(LiquidNeuron, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt

        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        self.tau_param = nn.Parameter(torch.ones(hidden_size) * 1.0)

    def forward(self, x, h):
        pre_activation = F.linear(x, self.W_in) + F.linear(h, self.W_rec) + self.bias
        dh = self.dt * ((-h + torch.tanh(pre_activation)) / torch.abs(self.tau_param))
        h_new = h + dh
        return h_new


class LiquidTimeConstantNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dt=0.1):
        super(LiquidTimeConstantNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.liquid = LiquidNeuron(input_size, hidden_size, dt)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.liquid(x[:, t, :], h)
        out = self.fc(h)
        return out


class CNNMNIST(nn.Module):

    def __init__(self, num_classes=10):
        super(CNNMNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model, example_input):

    param_count = count_parameters(model)
    if not THOP_AVAILABLE:
        return None, param_count
    try:
        model.eval()
        flops, params_thop = thop_profile(model, inputs=(example_input, ), verbose=False)
        return flops, param_count
    except Exception as e:
        return None, param_count


def measure_inference_time(model, device, input_batch, n_warmup=10, n_runs=10):

    model.eval()
    input_batch = input_batch.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_batch)

    timings = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(input_batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()
            timings.append(t1-t0)
    mean = float(np.mean(timings))
    std = float(np.std(timings))
    return mean, std


def train_one_epoch(model, device, data_loader, optimizer, criterion):

    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        n += inputs.size(0)
    return total_loss / n, correct / n


def eval_model(model, device, data_loader, criterion=None):

    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion:
                total_loss += criterion(outputs, labels).item() * inputs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            n += inputs.size(0)
    loss = total_loss / n if criterion else None
    acc = correct / n
    return loss, acc


def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.view(28, 28)
    ])
    train_set = datasets.MNIST(root=args.data_root_directory, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root=args.data_root_directory, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256)

    model = LiquidTimeConstantNetwork(input_size=28, hidden_size=64, output_size=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        correct = 0
        for images, labels in train_loader:
            # print(images.shape)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_set)
        train_loss = total_loss / len(train_set)

        model.eval()
        test_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                test_correct += (preds == labels).sum().item()
        test_acc = test_correct / len(test_set)

        print(f"Epoch {epoch+1} : Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MNIST Classification')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ltc_hidden', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_experiment(args)


# # compare_models.py
# import time
# import argparse
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict

# # Optional FLOPS tool
# try:
#     from thop import profile as thop_profile
#     THOP_AVAILABLE = True
# except Exception:
#     THOP_AVAILABLE = False

# # -------------------------
# # Models
# # -------------------------
# class LiquidNeuron(nn.Module):
#     def __init__(self, input_size, hidden_size, dt=0.1):
#         super(LiquidNeuron, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.dt = dt
#         self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
#         self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.tau_param = nn.Parameter(torch.ones(hidden_size) * 1.0)

#     def forward(self, x, h):
#         # x: (batch, input_size), h: (batch, hidden_size)
#         pre_activation = F.linear(x, self.W_in) + F.linear(h, self.W_rec) + self.bias
#         # ensure tau positive
#         tau = torch.abs(self.tau_param) + 1e-6
#         dh = self.dt * ((-h + torch.tanh(pre_activation)) / tau)
#         h_new = h + dh
#         return h_new

# class LiquidTimeConstantNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dt=0.1):
#         super(LiquidTimeConstantNetwork, self).__init__()
#         self.hidden_size = hidden_size
#         self.liquid = LiquidNeuron(input_size, hidden_size, dt)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # x shape: (batch, seq_len, input_size) -> for MNIST: seq_len=28 rows, input_size=28 pixels
#         batch_size, seq_len, input_size = x.shape
#         h = torch.zeros(batch_size, self.hidden_size, device=x.device)
#         for t in range(seq_len):
#             h = self.liquid(x[:, t, :], h)
#         out = self.fc(h)
#         return out

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SimpleCNN, self).__init__()
#         # small CNN baseline
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 28x28 -> 28x28
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28x28 -> 28x28
#         self.pool = nn.MaxPool2d(2, 2)               # 28 -> 14
#         self.fc1 = nn.Linear(64 * 14 * 14, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.act = nn.ReLU()

#     def forward(self, x):
#         # x: (batch,1,28,28)
#         x = self.act(self.conv1(x))
#         x = self.pool(self.act(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = self.act(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # -------------------------
# # Utilities: seed, params, flops, timing
# # -------------------------
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def compute_flops(model, device, example_input):
#     """
#     Uses thop if available. Returns flops (MACs) and params (also returns param count).
#     If thop not available, returns (None, param_count).
#     Note: thop returns FLOPs as raw number (multiply-add counted as 2 FLOPs in some configs).
#     """
#     param_count = count_parameters(model)
#     if not THOP_AVAILABLE:
#         return None, param_count
#     try:
#         model.eval()
#         # thop expects the input to match forward signature: pass tuple
#         flops, params_thop = thop_profile(model, inputs=(example_input,), verbose=False)
#         return flops, param_count
#     except Exception as e:
#         # Thop may fail on custom loops (like sequential over timesteps). return None.
#         return None, param_count

# def measure_inference_time(model, device, input_batch, n_warmup=10, n_runs=50):
#     model.eval()
#     input_batch = input_batch.to(device)
#     # warmup
#     with torch.no_grad():
#         for _ in range(n_warmup):
#             _ = model(input_batch)
#     # timed runs
#     timings = []
#     with torch.no_grad():
#         for _ in range(n_runs):
#             if device.type == 'cuda':
#                 torch.cuda.synchronize()
#             t0 = time.time()
#             _ = model(input_batch)
#             if device.type == 'cuda':
#                 torch.cuda.synchronize()
#             t1 = time.time()
#             timings.append(t1 - t0)
#     mean = float(np.mean(timings))
#     std = float(np.std(timings))
#     return mean, std

# # -------------------------
# # Training and evaluation
# # -------------------------
# def train_one_epoch(model, device, data_loader, optimizer, criterion):
#     model.train()
#     total_loss = 0.0
#     correct = 0
#     n = 0
#     for inputs, labels in data_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * inputs.size(0)
#         preds = outputs.argmax(1)
#         correct += (preds == labels).sum().item()
#         n += inputs.size(0)
#     return total_loss / n, correct / n

# def eval_model(model, device, data_loader, criterion=None):
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     n = 0
#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             if criterion:
#                 total_loss += criterion(outputs, labels).item() * inputs.size(0)
#             preds = outputs.argmax(1)
#             correct += (preds == labels).sum().item()
#             n += inputs.size(0)
#     loss = total_loss / n if criterion else None
#     acc = correct / n
#     return loss, acc

# # -------------------------
# # Dataset wrappers so both models see same images
# # -------------------------
# class RowSequenceTransform:
#     """Transform an MNIST image to (seq_len=28, input_size=28) float tensor."""
#     def __call__(self, x):
#         # x is (1,28,28) from ToTensor
#         # return (28,28) as float tensor
#         return x.squeeze(0).permute(0,1)  # shape (28,28)

# def ToImageTensor():
#     """Return a callable that keeps (1,28,28) for CNN"""
#     def fn(x):
#         return x  # ToTensor already gives (1,28,28)
#     return fn

# # -------------------------
# # Main experiment runner
# # -------------------------
# def run_experiment(args):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     set_seed(args.seed)

#     # Transforms
#     base_transform = transforms.ToTensor()  # yields (1,28,28)
#     # For the LTC model we need (seq_len=28, input_size=28):
#     transform_ltc = transforms.Compose([base_transform, RowSequenceTransform()])
#     transform_cnn = transforms.Compose([base_transform, ToImageTensor()])

#     # Datasets
#     train_set_ltc = datasets.MNIST(root=args.data_root, train=True, transform=transform_ltc, download=True)
#     test_set_ltc  = datasets.MNIST(root=args.data_root, train=False, transform=transform_ltc)
#     train_loader_ltc = DataLoader(train_set_ltc, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
#     test_loader_ltc  = DataLoader(test_set_ltc,  batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)

#     # For CNN use separate dataset objects to preserve transform
#     train_set_cnn = datasets.MNIST(root=args.data_root, train=True, transform=transform_cnn, download=False)
#     test_set_cnn  = datasets.MNIST(root=args.data_root, train=False, transform=transform_cnn)
#     train_loader_cnn = DataLoader(train_set_cnn, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
#     test_loader_cnn  = DataLoader(test_set_cnn,  batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)

#     # Build models
#     ltc = LiquidTimeConstantNetwork(input_size=28, hidden_size=args.ltc_hidden, output_size=10).to(device)
#     cnn = SimpleCNN(num_classes=10).to(device)

#     # Shared training setup
#     criterion = nn.CrossEntropyLoss()
#     opt_ltc = torch.optim.Adam(ltc.parameters(), lr=args.lr)
#     opt_cnn = torch.optim.Adam(cnn.parameters(), lr=args.lr)

#     # Logging storage
#     history = {
#         'ltc': {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]},
#         'cnn': {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]},
#     }

#     # Pre-compute param counts and FLOPS
#     print("Counting parameters and computing FLOPS (if thop available)...")
#     # Example inputs for flops
#     dummy_cnn = torch.randn(1,1,28,28).to(device)
#     dummy_ltc = torch.randn(1,28,28).to(device)  # LTC expects (batch, seq_len, input_size)
#     flops_cnn, params_cnn = compute_flops(cnn, device, dummy_cnn)
#     flops_ltc, params_ltc = compute_flops(ltc, device, dummy_ltc)
#     print(f"CNN params: {params_cnn:,}, CNN flops: {flops_cnn if flops_cnn else 'unavailable'}")
#     print(f"LTC params: {params_ltc:,}, LTC flops: {flops_ltc if flops_ltc else 'unavailable'}")

#     # Training loop: alternate training both models for fair GPU usage (or train sequentially)
#     for epoch in range(1, args.epochs + 1):
#         # Train LTC
#         t0 = time.time()
#         train_loss, train_acc = train_one_epoch(ltc, device, train_loader_ltc, opt_ltc, criterion)
#         val_loss, val_acc = eval_model(ltc, device, test_loader_ltc, criterion)
#         t1 = time.time()
#         history['ltc']['train_loss'].append(train_loss)
#         history['ltc']['train_acc'].append(train_acc)
#         history['ltc']['val_loss'].append(val_loss)
#         history['ltc']['val_acc'].append(val_acc)
#         print(f"[Epoch {epoch}] LTC - train loss {train_loss:.4f} acc {train_acc:.4f} | val acc {val_acc:.4f} | epoch time {t1-t0:.2f}s")

#         # Train CNN
#         t0 = time.time()
#         train_loss, train_acc = train_one_epoch(cnn, device, train_loader_cnn, opt_cnn, criterion)
#         val_loss, val_acc = eval_model(cnn, device, test_loader_cnn, criterion)
#         t1 = time.time()
#         history['cnn']['train_loss'].append(train_loss)
#         history['cnn']['train_acc'].append(train_acc)
#         history['cnn']['val_loss'].append(val_loss)
#         history['cnn']['val_acc'].append(val_acc)
#         print(f"[Epoch {epoch}] CNN - train loss {train_loss:.4f} acc {train_acc:.4f} | val acc {val_acc:.4f} | epoch time {t1-t0:.2f}s")

#     # After training: measure inference latency on a batch
#     # pick a single batch from test loader
#     sample_batch_ltc, _ = next(iter(test_loader_ltc))
#     sample_batch_cnn, _ = next(iter(test_loader_cnn))
#     # move to device but keep shapes expected by models
#     mean_ltc, std_ltc = measure_inference_time(ltc, device, sample_batch_ltc, n_warmup=20, n_runs=100)
#     mean_cnn, std_cnn = measure_inference_time(cnn, device, sample_batch_cnn, n_warmup=20, n_runs=100)

#     # Convert batch time to per-sample if needed
#     batch_size_ltc = sample_batch_ltc.size(0)
#     batch_size_cnn = sample_batch_cnn.size(0)
#     per_sample_ltc = mean_ltc / batch_size_ltc
#     per_sample_cnn = mean_cnn / batch_size_cnn

#     print("\n=== SUMMARY ===")
#     print(f"CNN parameters: {params_cnn:,}")
#     print(f"LTC parameters: {params_ltc:,}")
#     if flops_cnn: print(f"CNN FLOPS (approx): {flops_cnn:,}")
#     else: print("CNN FLOPS: unavailable (thop not installed or failed)")
#     if flops_ltc: print(f"LTC FLOPS (approx): {flops_ltc:,}")
#     else: print("LTC FLOPS: unavailable (thop not installed or failed)")
#     print(f"CNN inference (batch time mean): {mean_cnn:.6f}s ± {std_cnn:.6f}, per sample: {per_sample_cnn:.6f}s")
#     print(f"LTC inference (batch time mean): {mean_ltc:.6f}s ± {std_ltc:.6f}, per sample: {per_sample_ltc:.6f}s")

#     # Convergence: find epoch at which val_acc crosses thresholds
#     def epoch_to_reach(acc_list, threshold):
#         for i, a in enumerate(acc_list):
#             if a >= threshold:
#                 return i+1
#         return None

#     thresholds = [0.90, 0.95, 0.98]
#     for t in thresholds:
#         e_cnn = epoch_to_reach(history['cnn']['val_acc'], t)
#         e_ltc = epoch_to_reach(history['ltc']['val_acc'], t)
#         print(f"Epoch to reach {int(t*100)}% on val set -> CNN: {e_cnn}, LTC: {e_ltc}")

#     # Plot training curves
#     epochs = range(1, args.epochs+1)
#     plt.figure(figsize=(8,4))
#     plt.plot(epochs, history['cnn']['val_acc'], label='CNN val acc')
#     plt.plot(epochs, history['ltc']['val_acc'], label='LTC val acc')
#     plt.xlabel('Epoch')
#     plt.ylabel('Validation Accuracy')
#     plt.title('Validation Accuracy per Epoch')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('val_accuracy_comparison.png')
#     print("Saved validation accuracy plot to val_accuracy_comparison.png")

#     # Save history to .npz
#     np.savez('history_comparison.npz',
#              cnn_train_loss=np.array(history['cnn']['train_loss']),
#              cnn_val_acc=np.array(history['cnn']['val_acc']),
#              ltc_train_loss=np.array(history['ltc']['train_loss']),
#              ltc_val_acc=np.array(history['ltc']['val_acc']))
#     print("Saved training history to history_comparison.npz")

# # -------------------------
# # CLI
# # -------------------------
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, default='./data')
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--batch-size', type=int, default=128)
#     parser.add_argument('--eval-batch-size', type=int, default=256)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--ltc-hidden', type=int, default=64)
#     parser.add_argument('--seed', type=int, default=42)
#     args = parser.parse_args()
#     run_experiment(args)
