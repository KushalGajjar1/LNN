import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import subprocess
import os
import shutil

# --- Build the CUDA extension ---
def build_extension():
    """Builds the CUDA extension using setup.py."""
    print("Building CUDA extension (cuBLAS version)...")
    try:
        # Clean previous builds
        if os.path.exists('build'):
            shutil.rmtree('build')
        if os.path.exists('liquid_cuda.egg-info'):
            shutil.rmtree('liquid_cuda.egg-info')

        # Use pip for a more modern and robust build process
        subprocess.run(
            ['pip', 'install', '.'],
            capture_output=True,
            text=True,
            check=True
        )
        print("Build successful!")
    except subprocess.CalledProcessError as e:
        print("--- BUILD FAILED ---")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        exit()

# build_extension()

try:
    import liquid_cuda
except ImportError as e:
    print(f"\n\nFailed to import 'liquid_cuda': {e}")
    print("Please check the build output above for errors.")
    exit()

# --- Original PyTorch Implementation ---
class LiquidNeuron(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.1):
        super(LiquidNeuron, self).__init__()
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.tau_param = nn.Parameter(torch.ones(hidden_size) * 1.0)
        self.dt = dt

    def forward(self, x, h):
        pre_activation = F.linear(x, self.W_in) + F.linear(h, self.W_rec) + self.bias
        tau = torch.abs(self.tau_param).clamp(min=1e-6)
        dh = self.dt * ((-h + torch.tanh(pre_activation)) / tau)
        return h + dh

class LiquidTimeConstantNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt=0.1):
        super(LiquidTimeConstantNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.liquid = LiquidNeuron(input_size, hidden_size, dt)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        for t in range(x.size(1)):
            h = self.liquid(x[:, t, :], h)
        return self.fc(h)

# --- Custom CUDA Implementation (cuBLAS version) ---
class LiquidFunctionCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h_init, W_in, W_rec, bias, tau_param, dt):
        outputs, pre_activations, _ = liquid_cuda.forward(
            x, h_init, W_in, W_rec, bias, tau_param, dt)
        
        ctx.save_for_backward(x, h_init, W_in, W_rec, tau_param, outputs, pre_activations)
        ctx.dt = dt
        return outputs[-1]

    @staticmethod
    def backward(ctx, grad_h_final):
        x, h_init, W_in, W_rec, tau_param, outputs, pre_activations = ctx.saved_tensors
        
        seq_len = x.size(1)
        full_grad_out = torch.zeros_like(outputs)
        full_grad_out[-1] = grad_h_final.contiguous()

        grad_x, grad_h_init, grad_W_in, grad_W_rec, grad_bias, grad_tau_param = \
            liquid_cuda.backward(full_grad_out, x, h_init, W_in, W_rec, tau_param, 
                                 outputs, pre_activations, ctx.dt)
        
        return grad_x, grad_h_init, grad_W_in, grad_W_rec, grad_bias, grad_tau_param, None


class LiquidNeuronCUDA(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.1):
        super(LiquidNeuronCUDA, self).__init__()
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.tau_param = nn.Parameter(torch.ones(hidden_size) * 1.0)
        self.dt = dt

    def forward(self, x, h_init):
        return LiquidFunctionCUDA.apply(x, h_init, self.W_in, self.W_rec, self.bias, self.tau_param, self.dt)

class LiquidTimeConstantNetworkCUDA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt=0.1):
        super(LiquidTimeConstantNetworkCUDA, self).__init__()
        self.hidden_size = hidden_size
        self.liquid = LiquidNeuronCUDA(input_size, hidden_size, dt)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_init = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        h_final = self.liquid(x, h_init)
        return self.fc(h_final)

# --- Main Test Execution ---
def run_test():
    model_params = {'input_size': 64, 'hidden_size': 512, 'output_size': 10, 'dt': 0.1}
    data_params = {'seq_len': 100, 'batch_size': 64}
    
    print(f"\n--- Running Test with Model params: {model_params} ---")
    print(f"--- Data params: {data_params} ---")

    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return
    
    device = torch.device("cuda")
    torch.manual_seed(42)
    x = torch.randn(data_params['batch_size'], data_params['seq_len'], model_params['input_size'], device=device)

    torch.manual_seed(42)
    model_pytorch = LiquidTimeConstantNetwork(**model_params).to(device)
    
    torch.manual_seed(42)
    model_cuda = LiquidTimeConstantNetworkCUDA(**model_params).to(device)

    with torch.no_grad():
        model_cuda.liquid.W_in.copy_(model_pytorch.liquid.W_in)
        model_cuda.liquid.W_rec.copy_(model_pytorch.liquid.W_rec)
        model_cuda.liquid.bias.copy_(model_pytorch.liquid.bias)
        model_cuda.liquid.tau_param.copy_(model_pytorch.liquid.tau_param)
        model_cuda.fc.weight.copy_(model_pytorch.fc.weight)
        model_cuda.fc.bias.copy_(model_pytorch.fc.bias)

    print("\n--- Verifying Correctness ---")
    out_pytorch = model_pytorch(x)
    out_cuda = model_cuda(x)

    # --- UPDATED: Always print forward pass differences ---
    forward_pass_correct = torch.allclose(out_pytorch, out_cuda, atol=1e-4, rtol=1e-4)
    max_diff_fwd = torch.abs(out_pytorch - out_cuda).max().item()
    mean_diff_fwd = torch.abs(out_pytorch - out_cuda).mean().item()
    print(f"Forward Pass Correct: {forward_pass_correct}")
    print(f"  Max absolute difference (Forward): {max_diff_fwd:.6f}")
    print(f"  Mean absolute difference (Forward): {mean_diff_fwd:.6f}")

    grad_target = torch.randn_like(out_pytorch)
    
    model_pytorch.zero_grad()
    model_cuda.zero_grad()
    
    out_pytorch.backward(grad_target)
    out_cuda.backward(grad_target.clone())

    print("\n--- Backward Pass Verification ---")
    atol, rtol = 5e-2, 5e-2 
    
    # --- UPDATED: Always print backward pass differences ---
    for name, p_torch in model_pytorch.liquid.named_parameters():
        p_cuda = getattr(model_cuda.liquid, name)
        grad_correct = torch.allclose(p_torch.grad, p_cuda.grad, atol=atol, rtol=rtol)
        max_diff_bwd = torch.abs(p_torch.grad - p_cuda.grad).max().item()
        mean_diff_bwd = torch.abs(p_torch.grad - p_cuda.grad).mean().item()
        
        print(f"Backward Pass ({name}.grad) Correct: {grad_correct}")
        print(f"  Max absolute difference for {name}.grad: {max_diff_bwd:.6f}")
        print(f"  Mean absolute difference for {name}.grad: {mean_diff_bwd:.6f}")

    print("\n--- Running Benchmark ---")
    n_iterations = 100
    
    for _ in range(10): # Warmup
        model_pytorch(x).sum().backward()
        model_cuda(x).sum().backward()

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(n_iterations):
        model_pytorch(x).sum().backward()
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / n_iterations

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(n_iterations):
        model_cuda(x).sum().backward()
    torch.cuda.synchronize()
    cuda_time = (time.time() - start_time) / n_iterations

    print(f"PyTorch version took: {pytorch_time * 1000:.4f} ms per iteration")
    print(f"CUDA (cuBLAS) took:  {cuda_time * 1000:.4f} ms per iteration")
    if cuda_time > 0:
        print(f"Speedup: {pytorch_time / cuda_time:.2f}x")

if __name__ == '__main__':
    run_test()

