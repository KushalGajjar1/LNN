#include "liquid_cuda.h"
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(status) \
    do { \
        cudaError_t err = (status); \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error in " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

// --- KERNELS ---

__global__ void liquid_forward_elementwise_kernel(
    const float* __restrict__ pre_act_in,
    const float* __restrict__ pre_act_rec,
    const float* __restrict__ h,
    const float* __restrict__ bias,
    const float* __restrict__ tau_param,
    float* __restrict__ h_new,
    float* __restrict__ pre_activation_out,
    float* __restrict__ tanh_pre_activation_out,
    const float dt,
    const int batch_size,
    const int hidden_size)
{
    int bid = blockIdx.x;
    int hid = blockIdx.y * blockDim.x + threadIdx.x;

    if (bid >= batch_size || hid >= hidden_size) return;

    int idx = bid * hidden_size + hid;

    float pre_act = pre_act_in[idx] + pre_act_rec[idx] + bias[hid];
    pre_activation_out[idx] = pre_act;

    float tanh_pre = tanhf(pre_act);
    tanh_pre_activation_out[idx] = tanh_pre;

    float h_old = h[idx];
    float tau = fmaxf(fabsf(tau_param[hid]), 1e-6f);
    float dh = dt * ((-h_old + tanh_pre) / tau);
    h_new[idx] = h_old + dh;
}

__global__ void liquid_backward_elementwise_kernel(
    const float* __restrict__ grad_h_new,
    const float* __restrict__ h_prev,
    const float* __restrict__ pre_activation,
    const float* __restrict__ tau_param,
    float* __restrict__ grad_pre,
    float* __restrict__ grad_h_prev_from_ode,
    float* __restrict__ grad_bias, // <-- ADDED: Pass bias grad tensor
    float* __restrict__ grad_tau_param,
    const float dt,
    const int batch_size,
    const int hidden_size)
{
    int bid = blockIdx.x;
    int hid = blockIdx.y * blockDim.x + threadIdx.x;

    if (bid >= batch_size || hid >= hidden_size) return;

    int idx = bid * hidden_size + hid;

    float tau_raw = tau_param[hid];
    float tau = fmaxf(fabsf(tau_raw), 1e-6f);
    float pre_act_val = pre_activation[idx];
    float tanh_pre_act = tanhf(pre_act_val);
    const float g_hnew = grad_h_new[idx];

    float grad_tanh = 1.0f - tanh_pre_act * tanh_pre_act;
    float grad_pre_val = g_hnew * (dt / tau) * grad_tanh;
    grad_pre[idx] = grad_pre_val;
    
    // --- FIXED: Accumulate bias gradient here to match PyTorch ---
    // F.linear's bias gradient is the sum of the output gradient over the batch.
    // Here, grad_pre_val is the output gradient for this (batch, hidden) element.
    atomicAdd(&grad_bias[hid], grad_pre_val);
    
    grad_h_prev_from_ode[idx] = g_hnew * (1.0f - dt / tau);

    float grad_tau_contrib = g_hnew * (-dt) * (-h_prev[idx] + tanh_pre_act) / (tau * tau);
    if (tau_raw < 0.0f) grad_tau_contrib = -grad_tau_contrib;
    atomicAdd(&grad_tau_param[hid], grad_tau_contrib);
}


// --- cuBLAS Utilities & C++ WRAPPERS ---

cublasHandle_t get_cublas_handle() {
    static bool initialized = false;
    static cublasHandle_t handle;
    if (!initialized) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS handle creation failed");
        }
        initialized = true;
    }
    return handle;
}

#define CUBLAS_CHECK(status) \
    do { \
        cublasStatus_t err = (status); \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error " + std::to_string(err)); \
        } \
    } while(0)

void gemm_row_major(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                    int m, int n, int k, const float alpha, const float *A, const float *B,
                    const float beta, float *C) {
    int lda = (transa == CUBLAS_OP_N) ? k : m;
    int ldb = (transb == CUBLAS_OP_N) ? n : k;
    int ldc = n;
    CUBLAS_CHECK(cublasSgemm(handle, transb, transa, n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc));
}


std::vector<torch::Tensor> liquid_forward(
    torch::Tensor x, torch::Tensor h_init, torch::Tensor W_in, torch::Tensor W_rec,
    torch::Tensor bias, torch::Tensor tau_param, float dt) {

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto input_size = x.size(2);
    const auto hidden_size = h_init.size(1);
    
    auto h_t = h_init.clone();
    auto outputs = torch::zeros({seq_len, batch_size, hidden_size}, x.options());
    auto pre_activations = torch::zeros({seq_len, batch_size, hidden_size}, x.options());
    auto tanh_pre_activations = torch::zeros({seq_len, batch_size, hidden_size}, x.options());
    
    auto pre_act_in_buffer = torch::zeros({batch_size, hidden_size}, x.options());
    auto pre_act_rec_buffer = torch::zeros({batch_size, hidden_size}, x.options());

    cublasHandle_t handle = get_cublas_handle();

    const int threads_per_block = 256;
    dim3 threads(threads_per_block);
    dim3 blocks(batch_size, (hidden_size + threads_per_block - 1) / threads_per_block);

    for (int t = 0; t < seq_len; ++t) {
        auto x_t = x.slice(1, t, t + 1).squeeze(1).contiguous();
        auto h_new = torch::zeros_like(h_t);

        gemm_row_major(handle, CUBLAS_OP_N, CUBLAS_OP_T, batch_size, hidden_size, input_size, 1.0f,
                       x_t.data_ptr<float>(), W_in.data_ptr<float>(), 0.0f, pre_act_in_buffer.data_ptr<float>());
        
        gemm_row_major(handle, CUBLAS_OP_N, CUBLAS_OP_T, batch_size, hidden_size, hidden_size, 1.0f,
                       h_t.data_ptr<float>(), W_rec.data_ptr<float>(), 0.0f, pre_act_rec_buffer.data_ptr<float>());
        
        liquid_forward_elementwise_kernel<<<blocks, threads>>>(
            pre_act_in_buffer.data_ptr<float>(), pre_act_rec_buffer.data_ptr<float>(),
            h_t.data_ptr<float>(), bias.data_ptr<float>(), tau_param.data_ptr<float>(),
            h_new.data_ptr<float>(), pre_activations[t].data_ptr<float>(),
            tanh_pre_activations[t].data_ptr<float>(), dt, batch_size, hidden_size);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        h_t = h_new;
        outputs[t] = h_t;
    }
    return {outputs, pre_activations, tanh_pre_activations};
}

std::vector<torch::Tensor> liquid_backward(
    torch::Tensor grad_out, torch::Tensor x, torch::Tensor h_init, torch::Tensor W_in,
    torch::Tensor W_rec, torch::Tensor tau_param, torch::Tensor outputs,
    torch::Tensor pre_activations, float dt) {
    
    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto input_size = x.size(2);
    const auto hidden_size = h_init.size(1);

    auto grad_x = torch::zeros_like(x);
    auto grad_W_in = torch::zeros_like(W_in);
    auto grad_W_rec = torch::zeros_like(W_rec);
    auto grad_bias = torch::zeros_like(tau_param);
    auto grad_tau_param = torch::zeros_like(tau_param);

    auto grad_h_running = torch::zeros({batch_size, hidden_size}, x.options());
    auto grad_pre = torch::zeros({batch_size, hidden_size}, x.options());

    cublasHandle_t handle = get_cublas_handle();

    const int threads_per_block = 256;
    dim3 threads(threads_per_block);
    dim3 blocks(batch_size, (hidden_size + threads_per_block - 1) / threads_per_block);

    for (int t = seq_len - 1; t >= 0; --t) {
        auto total_grad_h_t = grad_out.slice(0, t, t + 1).squeeze(0) + grad_h_running;
        auto x_t = x.slice(1, t, t + 1).squeeze(1).contiguous();
        auto h_prev = (t == 0) ? h_init : outputs[t - 1];
        auto pre_act_t = pre_activations[t];
        
        auto grad_h_prev_ode = torch::zeros_like(h_init);
        
        liquid_backward_elementwise_kernel<<<blocks, threads>>>(
            total_grad_h_t.data_ptr<float>(), 
            h_prev.data_ptr<float>(), 
            pre_act_t.data_ptr<float>(),
            tau_param.data_ptr<float>(), 
            grad_pre.data_ptr<float>(),
            grad_h_prev_ode.data_ptr<float>(),
            grad_bias.data_ptr<float>(), // <-- Pass bias grad tensor
            grad_tau_param.data_ptr<float>(),
            dt, batch_size, hidden_size);

        gemm_row_major(handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden_size, input_size, batch_size, 1.0f,
                       grad_pre.data_ptr<float>(), x_t.data_ptr<float>(), 1.0f, grad_W_in.data_ptr<float>());

        gemm_row_major(handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden_size, hidden_size, batch_size, 1.0f,
                       grad_pre.data_ptr<float>(), h_prev.data_ptr<float>(), 1.0f, grad_W_rec.data_ptr<float>());
        
        auto grad_x_t = grad_x.slice(1, t, t + 1).squeeze(1);
        gemm_row_major(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, input_size, hidden_size, 1.0f,
                       grad_pre.data_ptr<float>(), W_in.data_ptr<float>(), 0.0f, grad_x_t.data_ptr<float>());
        
        auto grad_h_prev_rec = torch::zeros_like(h_init);
        gemm_row_major(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, hidden_size, hidden_size, 1.0f,
                       grad_pre.data_ptr<float>(), W_rec.data_ptr<float>(), 0.0f, grad_h_prev_rec.data_ptr<float>());

        grad_h_running = grad_h_prev_ode + grad_h_prev_rec;

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto grad_h_init = grad_h_running;
    return {grad_x, grad_h_init, grad_W_in, grad_W_rec, grad_bias, grad_tau_param};
}

