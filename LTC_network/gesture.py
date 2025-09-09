import numpy as np
import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import the custom cell implementations
from ltc_model import LTCCell, ODESolver
from ctrnn_models import CTRNN, NODE, CTGRU

def load_trace(filename):
    """Loads a single gesture CSV file."""
    df = pd.read_csv(filename, header=0)
    
    str_y = df["Phase"].values
    convert = {"D": 0, "P": 1, "S": 2, "H": 3, "R": 4}
    y = np.array([convert[s] for s in str_y], dtype=np.int64)
    
    x = df.values[:, :-1].astype(np.float32)
    return (x, y)

def cut_in_sequences(tup, seq_len, interleaved=False):
    """Cuts the data into sequences of a specified length."""
    x, y = tup
    num_sequences = x.shape[0] // seq_len
    sequences = []
    for s in range(num_sequences):
        start = seq_len * s
        end = start + seq_len
        sequences.append((x[start:end], y[start:end]))
        if interleaved and s < num_sequences - 1:
            start += seq_len // 2
            end = start + seq_len
            if end <= x.shape[0]:
                sequences.append((x[start:end], y[start:end]))
    return sequences

class GestureData:
    """Data handling class."""
    def __init__(self, seq_len=32, batch_size=16, device='cpu'):
        self.batch_size = batch_size
        self.device = device
        
        training_files = [
            "a3_va3.csv", "b1_va3.csv", "b3_va3.csv", "c1_va3.csv",
            "c3_va3.csv", "a2_va3.csv", "a1_va3.csv",
        ]
        
        train_traces = []
        for f in training_files:
            # You might need to adjust this path
            filepath = os.path.join("data/gesture", f)
            if not os.path.exists(filepath):
                print(f"Warning: Data file not found at {filepath}")
                print("Please ensure the 'data/gesture' directory is in your project root.")
                continue
            train_traces.extend(cut_in_sequences(load_trace(filepath), seq_len, interleaved=True))

        if not train_traces:
            raise FileNotFoundError("No training data was loaded. Check data paths.")

        train_x_np, train_y_np = list(zip(*train_traces))
        
        # Data is shaped as (num_sequences, seq_len, features)
        train_x_np = np.stack(train_x_np, axis=0)
        train_y_np = np.stack(train_y_np, axis=0)

        # Normalize features
        flat_x = train_x_np.reshape(-1, train_x_np.shape[-1])
        self.mean_x = np.mean(flat_x, axis=0)
        self.std_x = np.std(flat_x, axis=0)
        # Add a small epsilon to std_x to prevent division by zero
        self.std_x[self.std_x == 0] = 1e-8
        train_x_np = (train_x_np - self.mean_x) / self.std_x


        total_seqs = train_x_np.shape[0]
        print(f"Total number of training sequences: {total_seqs}")
        
        # Split data
        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)

        valid_indices = permutation[:valid_size]
        test_indices = permutation[valid_size:valid_size + test_size]
        train_indices = permutation[valid_size + test_size:]

        # Create torch tensors
        self.train_x = torch.from_numpy(train_x_np[train_indices]).to(device)
        self.train_y = torch.from_numpy(train_y_np[train_indices]).to(device)
        self.valid_x = torch.from_numpy(train_x_np[valid_indices]).to(device)
        self.valid_y = torch.from_numpy(train_y_np[valid_indices]).to(device)
        self.test_x = torch.from_numpy(train_x_np[test_indices]).to(device)
        self.test_y = torch.from_numpy(train_y_np[test_indices]).to(device)
        
    def get_dataloader(self, split='train'):
        if split == 'train':
            dataset = TensorDataset(self.train_x, self.train_y)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        elif split == 'valid':
            dataset = TensorDataset(self.valid_x, self.valid_y)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        else: # test
            dataset = TensorDataset(self.test_x, self.test_y)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

class GestureModel(nn.Module):
    """Wrapper for all recurrent models."""
    def __init__(self, input_size, model_type, model_size):
        super(GestureModel, self).__init__()
        self.model_type = model_type
        self.model_size = model_size
        
        # Instantiate the recurrent core
        if model_type == "lstm":
            self.rnn = nn.LSTM(input_size, model_size)
            rnn_output_size = self.rnn.hidden_size
        elif model_type.startswith("ltc"):
            solver = ODESolver.SemiImplicit
            if model_type.endswith("_rk"):
                solver = ODESolver.RungeKutta
            elif model_type.endswith("_ex"):
                solver = ODESolver.Explicit
            self.rnn = LTCCell(input_size, model_size, solver=solver)
            rnn_output_size = self.rnn.output_size
        elif model_type == "node":
            self.rnn = NODE(input_size, model_size, cell_clip=-1)
            rnn_output_size = self.rnn.output_size
        elif model_type == "ctgru":
            self.rnn = CTGRU(input_size, model_size, cell_clip=-1)
            rnn_output_size = self.rnn.output_size
        elif model_type == "ctrnn":
            self.rnn = CTRNN(input_size, model_size, cell_clip=-1, global_feedback=True)
            rnn_output_size = self.rnn.output_size
        else:
            raise ValueError(f"Unknown model type '{model_type}'")
        
        # The output layer
        self.fc = nn.Linear(rnn_output_size, 5) # 5 gesture classes

    def forward(self, x):
        # PyTorch RNNs expect input as (seq_len, batch, features)
        x = x.permute(1, 0, 2)
        
        if isinstance(self.rnn, nn.LSTM):
            outputs, _ = self.rnn(x)
        else:
            # Manual unrolling for custom cells
            batch_size = x.size(1)
            h = torch.zeros(batch_size, self.rnn.state_size).to(x.device)
            outputs = []
            for t in range(x.size(0)):
                output, h = self.rnn(x[t], h)
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0)

        # Pass RNN outputs to the fully connected layer
        # outputs shape: (seq_len, batch, hidden_size)
        logits = self.fc(outputs)
        return logits.permute(1, 2, 0) # (batch, classes, seq_len) for cross_entropy

def train(model, data, epochs, learning_rate, log_period):
    device = next(model.parameters()).device
    
    # LTC models may require a different learning rate
    lr = 0.01 if model.model_type.startswith("ltc") else learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare results logging
    results_dir = os.path.join("results", "gesture")
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"{model.model_type}_{model.model_size}.csv")
    if not os.path.isfile(result_file):
        with open(result_file, "w") as f:
            f.write("best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc\n")

    # Prepare model checkpointing
    checkpoint_dir = os.path.join("torch_sessions", "gesture")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model.model_type}.pth")

    best_valid_accuracy = 0
    best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
    
    train_loader = data.get_dataloader('train')
    
    for e in range(epochs):
        model.train()
        train_losses, train_accs = [], []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            logits = model(batch_x) # (batch, classes, seq_len)
            
            # Reshape for loss calculation: (N, C), (N)
            loss = criterion(logits, batch_y)
            
            # Check for NaN loss before backpropagation
            if torch.isnan(loss):
                print("Warning: NaN loss detected. Skipping backward pass for this batch.")
                continue

            loss.backward()
            
            # ** THE FIX: Add gradient clipping here **
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients to a max norm of 1.0

            optimizer.step()
            
            if model.model_type.startswith("ltc"):
                model.rnn.constrain_parameters()

            train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch_y).float().mean()
            train_accs.append(acc.item())

        if e % log_period == 0:
            model.eval()
            with torch.no_grad():
                # Validation
                valid_logits = model(data.valid_x)
                valid_loss = criterion(valid_logits, data.valid_y).item()
                valid_preds = torch.argmax(valid_logits, dim=1)
                valid_acc = (valid_preds == data.valid_y).float().mean().item()
                
                # Test
                test_logits = model(data.test_x)
                test_loss = criterion(test_logits, data.test_y).item()
                test_preds = torch.argmax(test_logits, dim=1)
                test_acc = (test_preds == data.test_y).float().mean().item()

            if valid_acc > best_valid_accuracy and e > 0:
                best_valid_accuracy = valid_acc
                torch.save(model.state_dict(), checkpoint_path)
                best_valid_stats = (
                    e,
                    np.mean(train_losses), np.mean(train_accs) * 100,
                    valid_loss, valid_acc * 100,
                    test_loss, test_acc * 100
                )
            
            print(f"Epoch {e:03d} | Train Loss: {np.mean(train_losses):.2f}, Acc: {np.mean(train_accs)*100:.2f}% | "
                  f"Valid Loss: {valid_loss:.2f}, Acc: {valid_acc*100:.2f}% | "
                  f"Test Loss: {test_loss:.2f}, Acc: {test_acc*100:.2f}%")
        
        if e > 0 and not np.isfinite(np.mean(train_losses)):
            print("Training diverged. Stopping.")
            break
            
    # Restore best model and print final results
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    best_epoch, tl, ta, vl, va, tsl, tsa = best_valid_stats
    print("\n" + "="*50)
    print(f"Best Epoch {best_epoch:03d} | Train Loss: {tl:.2f}, Acc: {ta:.2f}% | "
          f"Valid Loss: {vl:.2f}, Acc: {va:.2f}% | Test Loss: {tsl:.2f}, Acc: {tsa:.2f}%")
    print("="*50)

    with open(result_file, "a") as f:
        f.write(f"{best_epoch:03d},{tl:.2f},{ta:.2f},{vl:.2f},{va:.2f},{tsl:.2f},{tsa:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="ctrnn", help="Model type: lstm, ltc, ltc_rk, ltc_ex, node, ctgru, ctrnn")
    parser.add_argument('--log', default=1, type=int, help="Logging period in epochs")
    parser.add_argument('--size', default=32, type=int, help="Size of the recurrent hidden state")
    parser.add_argument('--epochs', default=200, type=int, help="Total number of epochs")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate (overridden by LTC models)")
    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data and model
    gesture_data = GestureData(device=device)
    model = GestureModel(input_size=32, model_type=args.model, model_size=args.size).to(device)
    
    print(f"Training model '{args.model}' with size {args.size} for {args.epochs} epochs.")
    train(model, gesture_data, epochs=args.epochs, learning_rate=args.lr, log_period=args.log)

