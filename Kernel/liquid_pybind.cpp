#include "liquid_cuda.h"

// This file now only contains the Pybind11 module definition.
// It includes the function declarations from the header file
// and exposes them to Python. The actual implementation, which
// contains CUDA code, is in liquid_cuda.cu and will be compiled
// by nvcc.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &liquid_forward, "Liquid Neuron forward (CUDA cuBLAS)");
    m.def("backward", &liquid_backward, "Liquid Neuron backward (CUDA cuBLAS)");
}
