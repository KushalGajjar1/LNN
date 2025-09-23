#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

// Device function for sigmoid
__device__ inline float sigmoid_device(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// CUDA Kernel using a double-buffer in shared memory for robust state updates
__global__ void ode_step_forward_kernel(
    float* v_pre_data,
    const float* w_num_sensory_data,
    const float* w_den_sensory_data,
    const float* W_data,
    const float* mu_data,
    const float* sigma_data,
    const float* erev_data,
    const float* cm_t_data,
    const float* gleak_data,
    const float* vleak_data,
    int batch_size,
    int num_units,
    int unfolds)
{
    // Shared memory will be used as a double buffer.
    // First half is for the current state, second half is for the next state.
    extern __shared__ float v_shared_buffer[];
    float* v_current = v_shared_buffer;
    float* v_next = v_shared_buffer + num_units;

    // Thread/Block identification
    const int batch_idx = blockIdx.x;
    const int unit_idx = threadIdx.x; // This is the target neuron 't'

    // Load initial state for the current unit into the 'v_current' buffer
    const int v_global_idx = batch_idx * num_units + unit_idx;
    v_current[unit_idx] = v_pre_data[v_global_idx];

    // Load parameters that are constant for this thread/unit
    const float w_num_sensory = w_num_sensory_data[v_global_idx];
    const float w_den_sensory = w_den_sensory_data[v_global_idx];
    const float cm_t = cm_t_data[unit_idx];
    const float gleak = gleak_data[unit_idx];
    const float vleak = vleak_data[unit_idx];

    // Synchronize to make sure all initial states are loaded into shared memory
    __syncthreads();

    // Main solver loop
    for (int i = 0; i < unfolds; ++i) {
        // Each thread reads the target neuron's state from the 'v_current' buffer
        const float v_target_current = v_current[unit_idx];
        
        float w_numerator_synapse = 0.0f;
        float w_denominator_synapse = 0.0f;

        // Reduction loop over the source neurons 's'
        for (int s = 0; s < num_units; ++s) {
            // Read the source neuron's state from the 'v_current' buffer
            const float v_source_current = v_current[s];

            // Parameters W, mu, etc., have (Source, Target) layout -> (s, t)
            const int param_idx = s * num_units + unit_idx;

            const float x = sigma_data[param_idx] * (v_source_current - mu_data[param_idx]);
            const float sig_val = sigmoid_device(x);

            const float w_act = W_data[param_idx] * sig_val;
            const float rev_act = w_act * erev_data[param_idx];

            w_numerator_synapse += rev_act;
            w_denominator_synapse += w_act;
        }
        
        const float total_w_num = w_numerator_synapse + w_num_sensory;
        const float total_w_den = w_denominator_synapse + w_den_sensory;

        const float numerator = cm_t * v_target_current + gleak * vleak + total_w_num;
        const float denominator = cm_t + gleak + total_w_den;
        
        // Write the calculated next state into the 'v_next' buffer
        v_next[unit_idx] = numerator / denominator;
        
        // Synchronize to ensure all threads have written to the 'v_next' buffer
        __syncthreads();

        // Copy the 'v_next' state back to 'v_current' for the next iteration
        v_current[unit_idx] = v_next[unit_idx];
        
        // Synchronize again to ensure the copy is complete before the next iteration begins
        __syncthreads();
    }

    // Write the final result from the 'v_current' buffer back to global memory
    v_pre_data[v_global_idx] = v_current[unit_idx];
}


// Kernel launcher function
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
    int unfolds)
{
    const int batch_size = v_pre.size(0);
    const int num_units = v_pre.size(1);

    // One thread per unit, one block per batch item
    const dim3 threads(num_units);
    const dim3 blocks(batch_size);
    // Allocate shared memory for two buffers (current and next state)
    const size_t shared_mem_size = 2 * num_units * sizeof(float);

    // Launch the kernel
    ode_step_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        v_pre.data_ptr<float>(),
        w_num_sensory.data_ptr<float>(),
        w_den_sensory.data_ptr<float>(),
        W.data_ptr<float>(),
        mu.data_ptr<float>(),
        sigma.data_ptr<float>(),
        erev.data_ptr<float>(),
        cm_t.data_ptr<float>(),
        gleak.data_ptr<float>(),
        vleak.data_ptr<float>(),
        batch_size,
        num_units,
        unfolds
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

