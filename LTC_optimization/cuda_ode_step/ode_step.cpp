#include <torch/extension.h>
#include <cstddef>

void ode_step_forward_cuda_launcher(
    torch::Tensor v_pre,
    const torch::Tensor& w_num_sensory,
    const torch::Tensor& w_den_sensory,
    const torch::Tensor& W,
    const torch::Tensor& mu,
    const torch::Tensor& sigma,
    const torch::Tensor& erev,
    const torch::Tensor& cm_t,
    const torch::Tensor& gleak,
    const torch::Tensor& vleak,
    int unfolds);

void ode_step_forward(
    torch::Tensor v_pre,
    const torch::Tensor& w_num_sensory,
    const torch::Tensor& w_den_sensory,
    const torch::Tensor& W,
    const torch::Tensor& mu,
    const torch::Tensor& sigma,
    const torch::Tensor& erev,
    const torch::Tensor& cm_t,
    const torch::Tensor& gleak,
    const torch::Tensor& vleak,
    int unfolds)
{
    TORCH_CHECK(v_pre.is_cuda(), "v_pre must be a CUDA tensor");
    TORCH_CHECK(v_pre.is_contiguous(), "v_pre must be contiguous");
    TORCH_CHECK(w_num_sensory.is_cuda(), "w_num_sensory must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");

    ode_step_forward_cuda_launcher(v_pre, w_num_sensory, w_den_sensory, W, mu, sigma, erev, cm_t, gleak, vleak, unfolds);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ode_step_forward, "LTC ODE Semi-Implicit Forward (CUDA)");
}

