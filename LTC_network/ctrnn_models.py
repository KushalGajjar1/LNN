import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class CTRNN(nn.Module):
    """
    PyTorch implementation of a Continuous-Time Recurrent Neural Network (CTRNN) cell.
    """
    def __init__(self, input_size, num_units, cell_clip=-1, global_feedback=False, fix_tau=True, tau=1.0, unfolds=6, delta_t=0.1):
        super(CTRNN, self).__init__()
        self._num_units = num_units
        self.global_feedback = global_feedback
        self.fix_tau = fix_tau
        self.cell_clip = cell_clip
        self._unfolds = unfolds
        self._delta_t = delta_t

        net_input_size = input_size
        if self.global_feedback:
            net_input_size += num_units
        self.net = nn.Linear(net_input_size, num_units)

        if self.fix_tau:
            self.register_buffer('tau', torch.tensor(tau))
        else:
            initial_tau_var = np.log(np.exp(tau) - 1.0)
            self.tau_var = nn.Parameter(torch.tensor(initial_tau_var, dtype=torch.float32))

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def forward(self, inputs, state):
        tau = F.softplus(self.tau_var) if not self.fix_tau else self.tau
        if not self.global_feedback:
            input_f_prime = torch.tanh(self.net(inputs))
        for _ in range(self._unfolds):
            if self.global_feedback:
                fused_input = torch.cat([inputs, state], dim=-1)
                input_f_prime = torch.tanh(self.net(fused_input))
            f_prime = -state / tau + input_f_prime
            state = state + self._delta_t * f_prime
            if self.cell_clip > 0:
                state = torch.clamp(state, -self.cell_clip, self.cell_clip)
        return state, state

class NODE(nn.Module):
    """
    PyTorch implementation of a Neural Ordinary Differential Equation (NODE) cell.
    """
    def __init__(self, input_size, num_units, cell_clip=-1, unfolds=6, delta_t=0.1):
        super(NODE, self).__init__()
        self._num_units = num_units
        self.cell_clip = cell_clip
        self._unfolds = unfolds
        self._delta_t = delta_t
        net_input_size = input_size + num_units
        self.net = nn.Linear(net_input_size, num_units)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _f_prime(self, inputs, state):
        fused_input = torch.cat([inputs, state], dim=-1)
        return torch.tanh(self.net(fused_input))

    def _ode_step_runge_kutta(self, inputs, state):
        for _ in range(self._unfolds):
            k1 = self._delta_t * self._f_prime(inputs, state)
            k2 = self._delta_t * self._f_prime(inputs, state + 0.5 * k1)
            k3 = self._delta_t * self._f_prime(inputs, state + 0.5 * k2)
            k4 = self._delta_t * self._f_prime(inputs, state + k3)
            state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            if self.cell_clip > 0:
                state = torch.clamp(state, -self.cell_clip, self.cell_clip)
        return state

    def forward(self, inputs, state):
        state = self._ode_step_runge_kutta(inputs, state)
        return state, state

class CTGRU(nn.Module):
    """
    PyTorch implementation of a Continuous-Time Gated Recurrent Unit (CT-GRU) cell.
    """
    def __init__(self, input_size, num_units, M=8, cell_clip=-1):
        super(CTGRU, self).__init__()
        self._input_size = input_size
        self._num_units = num_units
        self.M = M
        self.cell_clip = cell_clip

        ln_tau_table = np.empty(self.M)
        tau = 1.0
        for i in range(self.M):
            ln_tau_table[i] = np.log(tau)
            tau *= (10.0**0.5)
        self.register_buffer('ln_tau_table', torch.from_numpy(ln_tau_table).float())
        self.register_buffer('decay_factor', torch.exp(-1.0 / self.ln_tau_table))

        fused_input_size = input_size + num_units
        self.tau_r_layer = nn.Linear(fused_input_size, num_units * M)
        self.tau_s_layer = nn.Linear(fused_input_size, num_units * M)
        self.q_layer = nn.Linear(fused_input_size, num_units)

    @property
    def state_size(self):
        return self._num_units * self.M

    @property
    def output_size(self):
        return self._num_units

    def forward(self, inputs, state):
        h_hat = state.view(-1, self._num_units, self.M)
        h = torch.sum(h_hat, dim=2)
        fused_input = torch.cat([inputs, h], dim=-1)

        ln_tau_r = self.tau_r_layer(fused_input).view(-1, self._num_units, self.M)
        sf_input_r = -torch.square(ln_tau_r - self.ln_tau_table)
        rki = F.softmax(sf_input_r, dim=2)

        q_input = torch.sum(rki * h_hat, dim=2)
        reset_value = torch.cat([inputs, q_input], dim=1)
        qk = torch.tanh(self.q_layer(reset_value)).unsqueeze(2)

        ln_tau_s = self.tau_s_layer(fused_input).view(-1, self._num_units, self.M)
        sf_input_s = -torch.square(ln_tau_s - self.ln_tau_table)
        ski = F.softmax(sf_input_s, dim=2)

        h_hat_next = ((1 - ski) * h_hat + ski * qk) * self.decay_factor

        if self.cell_clip > 0:
            h_hat_next = torch.clamp(h_hat_next, -self.cell_clip, self.cell_clip)

        h_next = torch.sum(h_hat_next, dim=2)
        h_hat_next_flat = h_hat_next.view(-1, self._num_units * self.M)
        return h_next, h_hat_next_flat
