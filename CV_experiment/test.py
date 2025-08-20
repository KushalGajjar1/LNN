import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# -- Model --
class LiquidNeuron(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt

        # parameters
        self.W_in = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_rec = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        # parameterize tau so it stays positive: tau = softplus(tau_param) + eps
        self.tau_param = nn.Parameter(torch.ones(hidden_size) * 1.0)

        # stabilizing normalization on hidden state
        self.layernorm = nn.LayerNorm(hidden_size)

        # init
        nn.init.xavier_uniform_(self.W_in)
        # recurrent should be smaller to avoid chaotic dynamics
        nn.init.xavier_uniform_(self.W_rec)
        with torch.no_grad():
            self.W_rec.mul_(0.5)

    def forward(self, x, h):
        # x: (batch, input_size), h: (batch, hidden)
        pre = F.linear(x, self.W_in) + F.linear(h, self.W_rec) + self.bias
        tau = F.softplus(self.tau_param) + 1e-3  # positive
        # compute dh with explicit timestep dt to stabilize integration
        dh = self.dt * ((-h + torch.tanh(pre)) / tau)
        h_new = h + dh
        # optional normalization to keep scale reasonable
        h_new = self.layernorm(h_new)
        return h_new


class LiquidTimeConstant(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.liquid = LiquidNeuron(input_size, hidden_size, dt=dt)
        self.fc = nn.Linear(hidden_size, output_size)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, input_size = x.shape
        device = x.device
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        for t in range(seq_len):
            h = self.liquid(x[:, t, :], h)
        out = self.fc(h)  # logits
        return out

# -- Data --
transform = transforms.Compose([
    transforms.ToTensor(),                 # (1,28,28)
    transforms.Lambda(lambda t: t.squeeze(0))  # -> (28,28) row-by-row
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# -- Training setup --
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LiquidTimeConstant(input_size=28, hidden_size=256, output_size=10, dt=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -- Train/Validate --
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

for epoch in range(1, 16):
    model.train()
    running_loss = 0.0
    batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        # gradient clipping to avoid explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        batches += 1
        pbar.set_postfix(loss=running_loss / batches)

    scheduler.step()

    train_loss = running_loss / batches
    test_acc = evaluate(model, test_loader)
    print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

