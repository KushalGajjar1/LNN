import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CTRNN(nn.Module):

    def __init__(self, num_units, cell_clip=-1, global_feedback=False, fix_tau=True, unfolds=6, delta_t=0.1, tau=1.0):
        super().__init__()

        self._num_units = num_units
        self._unfolds = unfolds
        self._delta_t = delta_t
        self.global_feedback = global_feedback
        self.fix_tau = fix_tau
        self.tau = float(tau)
        self.cell_clip = cell_clip

        self.step_input = nn.LazyLinear(self._num_units)
        self.step_fused = nn.LazyLinear(self._num_units)

        if not self.fix_tau:
            self._tau_param = nn.Parameter(torch.tensor(self.tau, dtype=torch.float32))
        else:
            self.register_buffer("_tau_param", torch.tensor(self.tau, dtype=torch.float32))
        
    def forward(self, inputs, state):

        if not self.fix_tau:
            tau = F.softplus(self._tau_param)
        else:
            tau = self._tau_param

        if not self.global_feedback:
            input_f_prime = torch.tanh(self.step_input(inputs))

        for _ in range(self._unfolds):
            if self.global_feedback:
                fused = torch.cat([inputs, state], dim=-1)
                input_f_prime = torch.tanh(self.step_fused(fused))

            f_prime = -state / tau + input_f_prime
            state = state + self._delta_t * f_prime

            if self.cell_clip > 0:
                state = state.clamp(-self.cell_clip, self.cell_clip)

        return state, state

    def zero_state(self, batch, device=None, dtype=torch.float32):
        device = device if device is not None else next(self.parameters()).device
        return torch.zeros(batch, self._num_units, device=device, dtype=dtype)

    def export_weights(self, dirname, output_layer=None):
        os.makedirs(dirname, exist_ok=True)
        lin = self.step_fused if self.global_feedback else self.step_input
        W = lin.weight.detach().cpu().numpy()
        b = lin.bias.detach().cpu().numpy()
        np.savetxt(os.path.join(dirname, "w.csv"), W)
        np.savetxt(os.path.join(dirname, "b.csv"), b)
        tau = F.softplus(self._tau_param).detach().cpu().numpy() if not self.fix_tau else np.array([self.tau])
        np.savetxt(os.path.join(dirname, "tau.csv"), tau)
        if output_layer is not None:
            ow = output_layer.weight.detach().cpu().numpy()
            ob = output_layer.bias.detach().cpu().numpy()
            np.savetxt(os.path.join(dirname, "output_w.csv"), ow)
            np.savetxt(os.path.join(dirname, "output_b.csv"), ob)


class NODE(nn.Module):

    def __init__(self, num_units, cell_clip=-1, unfolds=6, delta_t=0.1):
        super().__init__()

        self._num_units = num_units
        self._unfolds = unfolds
        self._delta_t = delta_t
        self.cell_clip = cell_clip

        self.step = nn.LazyLinear(self._num_units)

    def _f_prime(self, inputs, state):
        fused = torch.cat([inputs, state], dim=-1)
        return torch.tanh(self.step(fused))
    
    def _ode_step_runge_kutta(self, inputs, state):
        for _ in range(self._unfolds):
            k1 = self._delta_t * self._f_prime(inputs, state)
            k2 = self._delta_t * self._f_prime(inputs, state + 0.5 * k1)
            k3 = self._delta_t * self._f_prime(inputs, state + 0.5 * k2)
            k4 = self._delta_t * self._f_prime(inputs, state + k3)

            state = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

            if self.cell_clip > 0:
                state = state.clamp(-self.cell_clip, self.cell_clip)

        return state
    
    def forward(self, inputs, state):
        state = self._ode_step_runge_kutta(inputs, state)
        return state, state
    
    def zero_state(self, batch, device=None, dtype=torch.float32):
        device = device if device is not None else next(self.parameters()).device
        return torch.zeros(batch, self._num_units, device=device, dtype=dtype)

    def export_weights(self, dirname, output_layer=None):
        os.makedirs(dirname, exist_ok=True)
        W = self.step.weight.detach().cpu().numpy()
        b = self.step.bias.detach().cpu().numpy()
        np.savetxt(os.path.join(dirname, "w.csv"), W)
        np.savetxt(os.path.join(dirname, "b.csv"), b)


# """
# PyTorch re-implementation of the TensorFlow CTRNN / NODE / CTGRU RNN cells
# from user's TF code.

# Each class is an nn.Module and exposes:
# - forward(inputs, state) -> (output, next_state)
# - zero_state(batch, device=None) -> initial state tensor
# - export_weights(dirname) to dump weights as .csv files (optional)

# Notes:
# - Uses LazyLinear so input size can be inferred on first forward call.
# - If you prefer explicit input sizes, replace LazyLinear with nn.Linear and
#   provide in_features at construction time.
# """

# import os
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class CTRNN(nn.Module):
#     """Continuous-time RNN cell (CTRNN) converted from the TF implementation.

#     forward(inputs, state) -> (state, state)
#     """

#     def __init__(self, num_units, cell_clip=-1, global_feedback=False, fix_tau=True, unfolds=6, delta_t=0.1, tau=1.0):
#         super().__init__()
#         self._num_units = num_units
#         self._unfolds = unfolds
#         self._delta_t = delta_t
#         self.global_feedback = global_feedback
#         self.fix_tau = fix_tau
#         self.tau = float(tau)
#         self.cell_clip = cell_clip

#         # Dense layers: use LazyLinear so we don't require input_size at construction.
#         # step_input used when global_feedback == False (input only)
#         # step_fused used when global_feedback == True (input + state)
#         self.step_input = nn.LazyLinear(self._num_units)
#         self.step_fused = nn.LazyLinear(self._num_units)

#         if not self.fix_tau:
#             # learnable log-parameter that is passed through softplus in forward
#             self._tau_param = nn.Parameter(torch.tensor(self.tau, dtype=torch.float32))
#         else:
#             self.register_buffer("_tau_param", torch.tensor(self.tau, dtype=torch.float32))

#     def forward(self, inputs, state):
#         # inputs: (batch, input_dim)
#         # state: (batch, num_units)
#         # returns (state, state) to match TF semantics
#         if not self.fix_tau:
#             tau = F.softplus(self._tau_param)
#         else:
#             tau = self._tau_param

#         # choose dense mapping
#         if not self.global_feedback:
#             input_f_prime = torch.tanh(self.step_input(inputs))

#         # run explicit Euler for _unfolds steps
#         for _ in range(self._unfolds):
#             if self.global_feedback:
#                 fused = torch.cat([inputs, state], dim=-1)
#                 input_f_prime = torch.tanh(self.step_fused(fused))

#             f_prime = -state / tau + input_f_prime
#             state = state + self._delta_t * f_prime

#             if self.cell_clip > 0:
#                 state = state.clamp(-self.cell_clip, self.cell_clip)

#         return state, state

#     def zero_state(self, batch, device=None, dtype=torch.float32):
#         device = device if device is not None else next(self.parameters()).device
#         return torch.zeros(batch, self._num_units, device=device, dtype=dtype)

#     def export_weights(self, dirname, output_layer=None):
#         os.makedirs(dirname, exist_ok=True)
#         # pick weights from the appropriate linear depending on global_feedback
#         lin = self.step_fused if self.global_feedback else self.step_input
#         W = lin.weight.detach().cpu().numpy()
#         b = lin.bias.detach().cpu().numpy()
#         np.savetxt(os.path.join(dirname, "w.csv"), W)
#         np.savetxt(os.path.join(dirname, "b.csv"), b)
#         tau = F.softplus(self._tau_param).detach().cpu().numpy() if not self.fix_tau else np.array([self.tau])
#         np.savetxt(os.path.join(dirname, "tau.csv"), tau)
#         if output_layer is not None:
#             ow = output_layer.weight.detach().cpu().numpy()
#             ob = output_layer.bias.detach().cpu().numpy()
#             np.savetxt(os.path.join(dirname, "output_w.csv"), ow)
#             np.savetxt(os.path.join(dirname, "output_b.csv"), ob)


# class NODE(nn.Module):
#     """Neural ODE style RNN cell using Runge-Kutta 4 integration.

#     forward(inputs, state) -> (state, state)
#     """

#     def __init__(self, num_units, cell_clip=-1, unfolds=6, delta_t=0.1):
#         super().__init__()
#         self._num_units = num_units
#         self._unfolds = unfolds
#         self._delta_t = delta_t
#         self.cell_clip = cell_clip

#         # fused input (inputs + state) -> num_units
#         self.step = nn.LazyLinear(self._num_units)

#     def _f_prime(self, inputs, state):
#         fused = torch.cat([inputs, state], dim=-1)
#         return torch.tanh(self.step(fused))

#     def _ode_step_runge_kutta(self, inputs, state):
#         for _ in range(self._unfolds):
#             k1 = self._delta_t * self._f_prime(inputs, state)
#             k2 = self._delta_t * self._f_prime(inputs, state + 0.5 * k1)
#             k3 = self._delta_t * self._f_prime(inputs, state + 0.5 * k2)
#             k4 = self._delta_t * self._f_prime(inputs, state + k3)

#             state = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

#             if self.cell_clip > 0:
#                 state = state.clamp(-self.cell_clip, self.cell_clip)

#         return state

#     def forward(self, inputs, state):
#         state = self._ode_step_runge_kutta(inputs, state)
#         return state, state

#     def zero_state(self, batch, device=None, dtype=torch.float32):
#         device = device if device is not None else next(self.parameters()).device
#         return torch.zeros(batch, self._num_units, device=device, dtype=dtype)

#     def export_weights(self, dirname, output_layer=None):
#         os.makedirs(dirname, exist_ok=True)
#         W = self.step.weight.detach().cpu().numpy()
#         b = self.step.bias.detach().cpu().numpy()
#         np.savetxt(os.path.join(dirname, "w.csv"), W)
#         np.savetxt(os.path.join(dirname, "b.csv"), b)


# class CTGRU(nn.Module):
#     """Continuous-time GRU (CT-GRU) converted from the TF code.

#     forward(inputs, state) -> (h_next, h_hat_next_flat)
#     where state is flattened (batch, num_units * M)
#     """

#     def __init__(self, num_units, M=8, cell_clip=-1):
#         super().__init__()
#         self._num_units = num_units
#         self.M = M
#         self.cell_clip = cell_clip

#         # Precompute ln_tau_table and register as buffer so it moves with the module
#         ln_tau_table = np.empty(self.M, dtype=np.float32)
#         tau = 1.0
#         for i in range(self.M):
#             ln_tau_table[i] = math.log(tau)
#             tau = tau * (10.0 ** 0.5)
#         self.register_buffer("ln_tau_table", torch.tensor(ln_tau_table, dtype=torch.float32))

#         # Dense layers used inside forward; sizes are lazy
#         self.tau_r = nn.LazyLinear(self._num_units * self.M)
#         self.tau_s = nn.LazyLinear(self._num_units * self.M)
#         self.detect_signal = nn.LazyLinear(self._num_units)

#     @property
#     def state_size(self):
#         return self._num_units * self.M

#     @property
#     def output_size(self):
#         return self._num_units

#     def forward(self, inputs, state):
#         # inputs expected shape: (batch, input_dim)
#         # state expected shape: (batch, num_units * M)

#         batch = inputs.shape[0]
#         # reshape state to (batch, num_units, M)
#         h_hat = state.view(batch, self._num_units, self.M)
#         h = torch.sum(h_hat, dim=2)  # (batch, num_units)

#         # fused input
#         fused_input = torch.cat([inputs, h], dim=1)

#         # tau_r: (batch, num_units * M) -> reshape to (batch, num_units, M)
#         ln_tau_r = self.tau_r(fused_input).view(batch, self._num_units, self.M)
#         sf_input_r = - (ln_tau_r - self.ln_tau_table.view(1, 1, -1)) ** 2
#         rki = F.softmax(sf_input_r, dim=2)

#         q_input = torch.sum(rki * h_hat, dim=2)  # (batch, num_units)
#         reset_value = torch.cat([inputs, q_input], dim=1)
#         qk = torch.tanh(self.detect_signal(reset_value))  # (batch, num_units)
#         qk = qk.view(batch, self._num_units, 1)

#         ln_tau_s = self.tau_s(fused_input).view(batch, self._num_units, self.M)
#         sf_input_s = - (ln_tau_s - self.ln_tau_table.view(1, 1, -1)) ** 2
#         ski = F.softmax(sf_input_s, dim=2)

#         # decay factor: exp(-1.0 / ln_tau_table)
#         # note: ln_tau_table contains log(tau). the original TF code used np.exp(-1.0/self.ln_tau_table)
#         # keep same semantics but protect against division by zero.
#         decay = torch.exp(-1.0 / (self.ln_tau_table + 1e-12)).view(1, 1, -1)

#         h_hat_next = ((1.0 - ski) * h_hat + ski * qk) * decay

#         if self.cell_clip > 0:
#             h_hat_next = h_hat_next.clamp(-self.cell_clip, self.cell_clip)

#         h_next = torch.sum(h_hat_next, dim=2)  # (batch, num_units)
#         h_hat_next_flat = h_hat_next.view(batch, self._num_units * self.M)

#         return h_next, h_hat_next_flat

#     def zero_state(self, batch, device=None, dtype=torch.float32):
#         device = device if device is not None else next(self.parameters()).device
#         return torch.zeros(batch, self._num_units * self.M, device=device, dtype=dtype)

#     def export_weights(self, dirname):
#         os.makedirs(dirname, exist_ok=True)
#         np.savetxt(os.path.join(dirname, "ln_tau_table.csv"), self.ln_tau_table.cpu().numpy())
#         # Save weights for tau_r, tau_s and detect_signal
#         for name, layer in [("tau_r", self.tau_r), ("tau_s", self.tau_s), ("detect_signal", self.detect_signal)]:
#             W = layer.weight.detach().cpu().numpy()
#             b = layer.bias.detach().cpu().numpy()
#             np.savetxt(os.path.join(dirname, f"{name}_w.csv"), W)
#             np.savetxt(os.path.join(dirname, f"{name}_b.csv"), b)


# # End of file
