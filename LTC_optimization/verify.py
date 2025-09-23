import torch
import time
import numpy as np
from ltc_model_optimized import LTCCell, ODESolver

def verify_and_benchmark(batch_size, input_size, num_units):
    print("Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Input Size: {input_size}")
    print(f"  Num Units:  {num_units}")
    print("------------------------------")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    # Initialize models
    print("Initializing models...")
    torch.manual_seed(42)
    np.random.seed(42) # FIX: Seed numpy to ensure identical model initializations
    model_torch = LTCCell(input_size, num_units, backend='torch').to(device)
    
    torch.manual_seed(42)
    np.random.seed(42) # FIX: Seed numpy to ensure identical model initializations
    model_triton = LTCCell(input_size, num_units, backend='triton').to(device)
    
    torch.manual_seed(42)
    np.random.seed(42) # FIX: Seed numpy to ensure identical model initializations
    model_cuda = LTCCell(input_size, num_units, backend='cuda').to(device)
    
    # Create some random input data
    inputs = torch.randn(batch_size, input_size).to(device)
    state = torch.randn(batch_size, num_units).to(device)

    # --- Correctness Check ---
    print("Running forward pass for correctness check...")
    with torch.no_grad():
        output_torch, _ = model_torch(inputs, state.clone())
        output_triton, _ = model_triton(inputs, state.clone())
        output_cuda, _ = model_cuda(inputs, state.clone())

    triton_correct = torch.allclose(output_torch, output_triton, atol=1e-5)
    triton_diff = torch.max(torch.abs(output_torch - output_triton))
    
    cuda_correct = torch.allclose(output_torch, output_cuda, atol=1e-5)
    cuda_diff = torch.max(torch.abs(output_torch - output_cuda))

    print("\n==============================")
    print("Verification Results")
    print("==============================")
    print(f"Triton implementation correct: {triton_correct} (Max diff: {triton_diff:.4e})")
    print(f"CUDA implementation correct:   {cuda_correct} (Max diff: {cuda_diff:.4e})")

    # --- Benchmarking ---
    warmup_iter = 10
    benchmark_iter = 100

    # PyTorch benchmark
    for _ in range(warmup_iter):
        model_torch(inputs, state)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(benchmark_iter):
        model_torch(inputs, state)
    torch.cuda.synchronize()
    end_time = time.time()
    torch_time = (end_time - start_time) * 1000 / benchmark_iter

    # Triton benchmark
    for _ in range(warmup_iter):
        model_triton(inputs, state)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(benchmark_iter):
        model_triton(inputs, state)
    torch.cuda.synchronize()
    end_time = time.time()
    triton_time = (end_time - start_time) * 1000 / benchmark_iter
    
    # CUDA benchmark
    for _ in range(warmup_iter):
        model_cuda(inputs, state)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(benchmark_iter):
        model_cuda(inputs, state)
    torch.cuda.synchronize()
    end_time = time.time()
    cuda_time = (end_time - start_time) * 1000 / benchmark_iter

    print("\n==============================")
    print("Benchmarking Results")
    print("==============================")
    print(f"PyTorch Backend: {torch_time:8.4f} ms")
    print(f"Triton Backend:  {triton_time:8.4f} ms")
    print(f"CUDA Backend:    {cuda_time:8.4f} ms")
    print("------------------------------")
    print(f"Triton Speedup vs PyTorch: {torch_time / triton_time:.2f}x")
    print(f"CUDA Speedup vs PyTorch:   {torch_time / cuda_time:.2f}x")
    print("==============================")


if __name__ == "__main__":
    verify_and_benchmark(batch_size=128, input_size=32, num_units=256)

