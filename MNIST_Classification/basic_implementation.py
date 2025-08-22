import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import argparse

# class LiquidNeuron(nn.Module):

#     def __init__(self, input_size, hidden_size):
#         super(LiquidNeuron, self).__init__()

#         self.input_size = input_size
#         self.hidden_size = hidden_size

#         self.W_in = nn.Parameter(torch.randn(hidden_size, input_size))
#         self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.tau = nn.Parameter(torch.ones(hidden_size))

#     def forward(self, x, h):
#         pre_activation = F.linear(x, self.W_in) + F.linear(h, self.W_rec) + self.bias
#         dh = (-h + torch.tanh(pre_activation)) / torch.abs(self.tau)
#         h_new = h + dh
#         return h_new


# class LiquidNeuralNetwork(nn.Module):

#     def __init__(self, input_size, hidden_size, output_size):
#         super(LiquidNeuralNetwork, self).__init__()

#         self.hidden_size = hidden_size
#         self.liquid = LiquidNeuron(input_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         batch_size, seq_len, input_size = x.shape
#         h = torch.zeros(batch_size, self.hidden_size, device=x.device)
#         for t in range(seq_len):
#             h = self.liquid(x[:, t, :], h)
#         out = self.fc(h)
#         return out

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


def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda t: t.squeeze(0)),
        lambda x: x.view(28, 28),
        # transforms.Normalize((0.5, ), (0.5, ))
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
    parser.add_argument('--data_root_directory', type=str, default='./data')
    args = parser.parse_args()

    train(args)