import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid(x):
    """JIT-compatible sigmoid function."""
    return 1.0 / (1.0 + tl.exp(-x))

@triton.jit
def single_ode_step_kernel(
    # Pointers to Tensors
    V_IN_ptr, V_OUT_ptr, W_NUM_SENSORY_ptr, W_DEN_SENSORY_ptr,
    W_ptr, MU_ptr, SIGMA_ptr, EREV_ptr,
    CM_T_ptr, GLEAK_ptr, VLEAK_ptr,
    # Tensor dimensions
    BATCH_SIZE, NUM_UNITS,
    # Strides for tensor indexing
    stride_v_in_batch, stride_v_in_units,
    stride_v_out_batch, stride_v_out_units,
    stride_w_num_batch, stride_w_num_units,
    stride_w_den_batch, stride_w_den_units,
    stride_W_u1, stride_W_u2,
    stride_MU_u1, stride_MU_u2,
    stride_SIGMA_u1, stride_SIGMA_u2,
    stride_EREV_u1, stride_EREV_u2,
    stride_cm_t, stride_gleak, stride_vleak,
    # Kernel parameters
    BLOCK_SIZE_N: tl.constexpr, # Tile size for the target dimension
    BLOCK_SIZE_M: tl.constexpr  # Tile size for the source dimension
):
    """
    Optimized Triton kernel for a SINGLE semi-implicit ODE solver step.
    This kernel is called multiple times by a Python loop to ensure synchronization.
    """
    # Program IDs for batch and target neuron block
    pid_batch = tl.program_id(0)
    pid_n = tl.program_id(1)

    # --- Create pointers for the current batch item ---
    v_in_batch_ptr = V_IN_ptr + pid_batch * stride_v_in_batch
    w_num_sensory_batch_ptr = W_NUM_SENSORY_ptr + pid_batch * stride_w_num_batch
    w_den_sensory_batch_ptr = W_DEN_SENSORY_ptr + pid_batch * stride_w_den_batch
    
    # --- Pointers for the current block of target neurons 't' ---
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Load constant parameters for this target block
    mask_n = offs_n < NUM_UNITS
    cm_t = tl.load(CM_T_ptr + offs_n * stride_cm_t, mask=mask_n)
    gleak = tl.load(GLEAK_ptr + offs_n * stride_gleak, mask=mask_n)
    vleak = tl.load(VLEAK_ptr + offs_n * stride_vleak, mask=mask_n)

    # Load current state and sensory inputs for this target block
    v_pre_in = tl.load(v_in_batch_ptr + offs_n * stride_v_in_units, mask=mask_n)
    w_num_sensory = tl.load(w_num_sensory_batch_ptr + offs_n * stride_w_num_units, mask=mask_n)
    w_den_sensory = tl.load(w_den_sensory_batch_ptr + offs_n * stride_w_den_units, mask=mask_n)
    
    # Accumulators for synapse contributions, initialized to zero
    acc_w_num_synapse = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    acc_w_den_synapse = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Loop over blocks of source neurons 's'
    for m in range(0, NUM_UNITS, BLOCK_SIZE_M):
        offs_m = m + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < NUM_UNITS
        
        # Load a block of source neuron states v_pre[s] from the input tensor
        v_s = tl.load(v_in_batch_ptr + offs_m * stride_v_in_units, mask=mask_m)

        # Load blocks/tiles of the parameter matrices [S_block, T_block]
        mu_ptrs = MU_ptr + offs_m[:, None] * stride_MU_u1 + offs_n[None, :] * stride_MU_u2
        sigma_ptrs = SIGMA_ptr + offs_m[:, None] * stride_SIGMA_u1 + offs_n[None, :] * stride_SIGMA_u2
        W_ptrs = W_ptr + offs_m[:, None] * stride_W_u1 + offs_n[None, :] * stride_W_u2
        erev_ptrs = EREV_ptr + offs_m[:, None] * stride_EREV_u1 + offs_n[None, :] * stride_EREV_u2
        
        mask_tile = (offs_m[:, None] < NUM_UNITS) & (offs_n[None, :] < NUM_UNITS)
        
        mu = tl.load(mu_ptrs, mask=mask_tile, other=0.0)
        sigma = tl.load(sigma_ptrs, mask=mask_tile, other=0.0)
        W = tl.load(W_ptrs, mask=mask_tile, other=0.0)
        erev = tl.load(erev_ptrs, mask=mask_tile, other=0.0)

        # Perform computation for the tile
        mues = v_s[:, None] - mu
        sig_vals = _sigmoid(sigma * mues)
        w_act = W * sig_vals
        rev_act = w_act * erev
        
        # Reduce over the source block dimension 's' and update accumulators
        acc_w_num_synapse += tl.sum(rev_act, axis=0)
        acc_w_den_synapse += tl.sum(w_act, axis=0)

    # Combine with sensory inputs after iterating through all source blocks
    total_w_num = acc_w_num_synapse + w_num_sensory
    total_w_den = acc_w_den_synapse + w_den_sensory

    # Update rule for the v_pre vector block
    numerator = cm_t * v_pre_in + gleak * vleak + total_w_num
    denominator = cm_t + gleak + total_w_den
    
    v_pre_out = numerator / denominator

    # Write the final state vector block to the output tensor
    v_out_batch_ptr = V_OUT_ptr + pid_batch * stride_v_out_batch
    tl.store(v_out_batch_ptr + offs_n * stride_v_out_units, v_pre_out, mask=mask_n)


def ode_step_forward_triton(
    v_pre, w_numerator_sensory, w_denominator_sensory,
    W, mu, sigma, erev,
    cm_t, gleak, vleak,
    unfolds=6
):
    """
    Python wrapper for the optimized Triton kernel.
    It iterates by calling the single-step kernel 'unfolds' times.
    """
    BATCH_SIZE, NUM_UNITS = v_pre.shape
    
    # Use a double-buffering strategy in Python
    v_in = v_pre # The initial state is the first input
    v_out = torch.empty_like(v_pre)

    # Heuristic for tile sizes
    BLOCK_SIZE_N = 32 if NUM_UNITS >= 64 else 16
    BLOCK_SIZE_M = 32 if NUM_UNITS >= 64 else 16

    # Kernel launch grid. Each program instance handles one BATCH and one TILE of target neurons.
    grid = (BATCH_SIZE, triton.cdiv(NUM_UNITS, BLOCK_SIZE_N))

    for i in range(unfolds):
        single_ode_step_kernel[grid](
            v_in, v_out, w_numerator_sensory, w_denominator_sensory,
            W, mu, sigma, erev,
            cm_t, gleak, vleak,
            BATCH_SIZE, NUM_UNITS,
            v_in.stride(0), v_in.stride(1),
            v_out.stride(0), v_out.stride(1),
            w_numerator_sensory.stride(0), w_numerator_sensory.stride(1),
            w_denominator_sensory.stride(0), w_denominator_sensory.stride(1),
            W.stride(0), W.stride(1),
            mu.stride(0), mu.stride(1),
            sigma.stride(0), sigma.stride(1),
            erev.stride(0), erev.stride(1),
            cm_t.stride(0), gleak.stride(0), vleak.stride(0),
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
        )
        # Swap buffers for the next iteration. The output of this step is the input to the next.
        v_in, v_out = v_out, v_in

    # The final result is in v_in because of the last swap
    return v_in

