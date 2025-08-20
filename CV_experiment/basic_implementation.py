import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class LiquidNeuron(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LiquidNeuron, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, h):
        pre_activation = F.linear(x, self.W_in) + F.linear(h, self.W_rec) + self.bias
        dh = (-h + torch.tanh(pre_activation)) / torch.abs(self.tau)
        h_new = h + dh
        return h_new
    

class LiquidTimeConstant(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidTimeConstant, self).__init__()

        self.hidden_size = hidden_size
        self.liquid = LiquidNeuron(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t  in range(seq_len):
            h = self.liquid(x[:, t, :], h)
        out = self.fc(h)
        return out
    

transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: x.view(28, 28)
])


train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

model = LiquidTimeConstant(input_size=28, hidden_size=64, output_size=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    print(f"Train loss : {loss.item()}")

model.eval()
test_correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        test_correct += (pred == y).sum().item()
test_acc = test_correct / len(test_set)
print(f"Test accuracy : {test_acc}")