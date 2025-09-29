import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch.autograd import Function
import time

polynomial_cuda_source = r"""

template<typename scalar_t>
__global__ void polynomial_activation_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ output, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        scalar_t val = x[idx];
        output[idx] = val * val + val + 1;
    }
}

torch::Tensor polynomial_activation_cuda(torch::Tensor x){
    auto output = torch::empty_like(x);
    int threads = 1024;
    int blocks = (x.numel() + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "polynomial_activation_cuda", ([&] {
        polynomial_activation_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            x.numel()
        );
    }));
    
    return output;
}

"""

polynomial_cuda_declaration = r"""
torch::Tensor polynomial_activation_cuda(torch::Tensor input);
"""

cuda_polynomial_module = load_inline(
    name="cuda_polynomial_module",
    cpp_sources=polynomial_cuda_declaration,
    cuda_sources=polynomial_cuda_source,
    functions=["polynomial_activation_cuda"],
    verbose=True,
    extra_cuda_cflags=["-std=c++17"]
)


class CUDAPolynomialActivation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return cuda_polynomial_module.polynomial_activation_cuda(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented")


class PolynomialActivation(nn.Module):

    def __init__(self, implementation="pytorch"):
        super().__init__()
        self.implementation = implementation

    def forward(self, x):
        if self.implementation == 'pytorch':
            return x**2 + x + 1
        elif self.implementation == 'cuda':
            return CUDAPolynomialActivation.apply(x)
        else:
            raise ValueError(f"Unknown implementation : {self.implementation}")
        

def benchmark(func, x, name, num_runs=1000):
    start_time = time.time()
    for _ in range(num_runs):
        func(x)
    torch.cuda.synchronize()
    end_time = time.time()
    return f"{name} : {(end_time - start_time) / num_runs * 1000:.4f} ms"


def main():

    torch.manual_seed(0)
    # x = torch.tensor([1, 2, 3], device='cuda')
    x = torch.randn(1000000, device='cuda')

    pytorch_activation = PolynomialActivation(implementation='pytorch').cuda()
    cuda_activation = PolynomialActivation(implementation='cuda').cuda()

    out = cuda_activation.forward(x)
    print(out)

    pytorch_time = benchmark(pytorch_activation, x, "PyTorch built-in")
    cuda_time = benchmark(cuda_activation, x, "CUDA extension")

    print(pytorch_time)
    print(cuda_time)


if __name__ == '__main__':

    main()