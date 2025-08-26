import os
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2


class LTCCell(nn.Module):

    def __init__(self, num_units, device=None, dtype=torch.float32):
        super().__init__()

        self.device = device if device else torch.device("cpu")
        self.dtype = dtype

        self._input_size = -1
        self._num_units = num_units
        self._is_built = False

        self._ode_solver_unfolds = 6
        self._solver = ODESolver.SemiImplicit

        self._input_mapping = MappingType.Affine

        self._erev_init_factor = 1.0

        self._w_init_max = 1.0
        self._w_init_min = 0.01
        self._cm_init_min = 0.5
        self._cm_init_max = 0.5
        self._gleak_init_min = 1.0
        self._gleak_init_max = 1.0

        self._w_min_value = 0.00001
        self._w_max_value = 1000
        self._gleak_min_value = 0.00001
        self._gleak_max_value = 1000
        self._cm_t_min_value = 0.000001
        self._cm_t_max_value = 1000

        self._fix_cm = None
        self._fix_gleak = None
        self._fix_vleak = None

        self._params_created = False

    @property
    def state_size(self):
        return self._num_units
    
    @property
    def output_size(self):
        return self._num_units

    def _map_inputs(self, inputs, reuse_scope=False):

        if self._input_mapping in (MappingType.Linear, MappingType.Affine):
            inputs = inputs * self.input_w.unsqueeze(0)
        if self._input_mapping is MappingType.Affine:
            inputs = inputs + self.input_b.unsqueeze(0)
        return inputs
        
    def _get_variables(self):

        assert self._input_size > 0, "Input size must be set before creating variables"

        self.sensory_mu = nn.Parameter(torch.empty(self._input_size, self._num_units, device=self.device, dtype=self.dtype).uniform_(0.3, 0.8))
        self.sensory_sigma = nn.Parameter(torch.empty(self._input_size, self._num_units, device=self.device, dtype=self.dtype).uniform_(3.0, 8.0))
        self.sensory_W = nn.Parameter(torch.from_numpy(np.random.uniform(low=self._w_init_min, high=self._w_init_max, size=(self._input_size, self._num_units))).to(dtype=self.dtype, device=self.device))

        sensory_erev_init = (2 * torch.randint(0, 2, (self._input_size, self._num_units), device=self.device, dtype=torch.int32) - 1).to(dtype=self.dtype)

        self.sensory_erev = nn.Parameter(sensory_erev_init * float(self._erev_init_factor))

        self.mu = nn.Parameter(torch.empty(self._num_units, self._num_units, device=self.device, dtype=self.dtype).uniform_(0.3, 0.8))
        self.sigma = nn.Parameter(torch.empty(self._num_units, self._num_units, device=self.device, dtype=self.dtype).uniform_(3.0, 8.0))

        self.W = nn.Parameter(torch.from_numpy(np.random.uniform(low=self._w_init_min, high=self._w_init_max, size=(self._num_units, self._num_units))).to(dtype=self.dtype, device=self.device))

        erev_init = (2 * torch.randint(0, 2, (self._num_units, self._num_units), device=self.device, dtype=torch.int32) - 1).to(dtype=self.dtype)
        self.erev = nn.Parameter(erev_init * float(self._erev_init_factor))

        if self._fix_vleak is None:
            self.vleak = nn.Parameter(torch.empty(self._num_units, device=self.device, dtype=self.dtype).uniform_(-0.2, 0.2))
        else:
            self.register_buffer("vleak", torch.tensor(self._fix_vleak, device=self.device, dtype=self.dtype))

        if self._fix_gleak is None:
            if self._gleak_init_max > self._gleak_init_min:
                gleak_val = torch.empty(self._num_units, device=self.device, dtype=self.dtype).uniform_(self._gleak_init_min, self._gleak_init_max)
            else:
                gleak_val = torch.full((self._num_units, ), fill_value=self._gleak_init_min, device=self.device, dtype=self.dtype)
            self.gleak = nn.Parameter(gleak_val)
        else:
            self.register_buffer("gleak", torch.tensor(self._fix_gleak, device=self.device, dtype=self.dtype))

        if self._fix_cm is None:
            if self._cm_init_max > self._cm_init_min:
                cm_val = torch.empty(self._num_units, device=self.device, dtype=self.dtype).uniform_(self._cm_init_min, self._cm_init_max)
            else:
                cm_val = torch.full((self._num_units_, ), fill_value=self._cm_init_min, device=self.device, dtype=self.dtype)
            self.cm_t = nn.Pramameter(cm_val)
        else:
            self.register_buffer("cm_t", torch.tensor(self._fix_cm, device=self.device, dtype=self.dtype))
        
        if self._input_mapping in (MappingType.Linear, MappingType.Affine):
            self.input_w = nn.Parameter(torch.ones(self._input_size, device=self.device, dtype=self.dtype))
        if self._input_mapping is MappingType.Affine:
            self.input_b = nn.Parameter(torch.zeros(self._input_size, device=self.device, dtype=self.dtype))

        self._params_created = True

    def forward(self, inputs, state):

        if not self._is_built:
            self._is_built = True
            self._input_size = int(inputs.shape[-1])
            self._get_variables()
        else:
            if self._input_size != int(inputs.shape[-1]):
                raise ValueError("You first feed an input with {} features and now one with {} features, that is not possible".format(self._input_size, int(inputs.shape[-1])))

        inputs = self._map_inputs(inputs)

        if self._solver == ODESolver.Explicit:
            next_state = self._ode_step_explicit(inputs, state, _ode_solver_unfolds=self._ode_solver_unfolds)
        elif self._solver == ODESolver.SemiImplicit:
            next_state = self._ode_step(inputs, state)
        elif self._solver == ODESolver.RungeKutta:
            next_state = self._ode_step_runge_kutta(inputs, state)
        else:
            raise ValueError("Unknown ODE solver '{}'".format(str(self._solver)))

        outputs = next_state
        return outputs, next_state

    def _sigmoid(self, v_pre, mu, sigma):

        v_pre_reshape = v_pre.unsqueeze(-1)
        mues = v_pre_reshaped - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_step(self, inputs, state):

        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev

        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(self._ode_solver_unfolds):

import os
# from enum import Enum

# import numpy as np
# import torch
# import torch.nn as nn


# class MappingType(Enum):
#     Identity = 0
#     Linear = 1
#     Affine = 2


# class ODESolver(Enum):
#     SemiImplicit = 0
#     Explicit = 1
#     RungeKutta = 2


# class LTCCell(nn.Module):
#     """
#     PyTorch port of the provided TensorFlow LTCCell.
#     Usage:
#         cell = LTCCell(num_units)
#         state = torch.zeros(batch_size, num_units, device=device)
#         out, next_state = cell(inputs, state)
#     """

#     def __init__(self, num_units, device=None, dtype=torch.float32):
#         super().__init__()
#         self.device = device or torch.device("cpu")
#         self.dtype = dtype

#         # config
#         self._input_size = -1
#         self._num_units = num_units
#         self._is_built = False

#         # ODE solver settings
#         self._ode_solver_unfolds = 6
#         self._solver = ODESolver.SemiImplicit

#         self._input_mapping = MappingType.Affine

#         self._erev_init_factor = 1.0

#         # initial ranges (match your TF code defaults)
#         self._w_init_max = 1.0
#         self._w_init_min = 0.01
#         self._cm_init_min = 0.5
#         self._cm_init_max = 0.5
#         self._gleak_init_min = 1.0
#         self._gleak_init_max = 1.0

#         # parameter clamp limits
#         self._w_min_value = 1e-5
#         self._w_max_value = 1e3
#         self._gleak_min_value = 1e-5
#         self._gleak_max_value = 1e3
#         self._cm_t_min_value = 1e-6
#         self._cm_t_max_value = 1e3

#         # optional fixed params (if set, stored as buffers and not trained)
#         self._fix_cm = None
#         self._fix_gleak = None
#         self._fix_vleak = None

#         # placeholders for parameters created in _get_variables()
#         # sensory_*, W, erev, mu, sigma, gleak, cm_t, vleak, etc.
#         # They will be created dynamically on first forward() call.
#         self._params_created = False

#     @property
#     def state_size(self):
#         return self._num_units

#     @property
#     def output_size(self):
#         return self._num_units

#     def _map_inputs(self, inputs, reuse_scope=False):
#         """
#         inputs: (batch, input_size)
#         If mapping is Linear or Affine, apply elementwise weight and optionally bias across last dimension.
#         """
#         if self._input_mapping in (MappingType.Linear, MappingType.Affine):
#             # input_w shape: (input_size,)
#             inputs = inputs * self.input_w.unsqueeze(0)  # broadcast over batch
#         if self._input_mapping is MappingType.Affine:
#             inputs = inputs + self.input_b.unsqueeze(0)
#         return inputs

#     def _get_variables(self):
#         """Create parameters / buffers (called lazily when the first input arrives)."""
#         assert self._input_size > 0, "Input size must be set before creating variables."

#         # sensory params (input_size x num_units)
#         self.sensory_mu = nn.Parameter(
#             torch.empty(self._input_size, self._num_units, device=self.device, dtype=self.dtype).uniform_(0.3, 0.8)
#         )
#         self.sensory_sigma = nn.Parameter(
#             torch.empty(self._input_size, self._num_units, device=self.device, dtype=self.dtype).uniform_(3.0, 8.0)
#         )
#         self.sensory_W = nn.Parameter(
#             torch.from_numpy(
#                 np.random.uniform(low=self._w_init_min, high=self._w_init_max, size=(self._input_size, self._num_units))
#             ).to(dtype=self.dtype, device=self.device)
#         )

#         sensory_erev_init = (2 * torch.randint(0, 2, (self._input_size, self._num_units), device=self.device, dtype=torch.int32) - 1).to(
#             dtype=self.dtype
#         )
#         self.sensory_erev = nn.Parameter(sensory_erev_init * float(self._erev_init_factor))

#         # recurrent params (num_units x num_units)
#         self.mu = nn.Parameter(
#             torch.empty(self._num_units, self._num_units, device=self.device, dtype=self.dtype).uniform_(0.3, 0.8)
#         )
#         self.sigma = nn.Parameter(
#             torch.empty(self._num_units, self._num_units, device=self.device, dtype=self.dtype).uniform_(3.0, 8.0)
#         )
#         self.W = nn.Parameter(
#             torch.from_numpy(
#                 np.random.uniform(low=self._w_init_min, high=self._w_init_max, size=(self._num_units, self._num_units))
#             ).to(dtype=self.dtype, device=self.device)
#         )

#         erev_init = (2 * torch.randint(0, 2, (self._num_units, self._num_units), device=self.device, dtype=torch.int32) - 1).to(
#             dtype=self.dtype
#         )
#         self.erev = nn.Parameter(erev_init * float(self._erev_init_factor))

#         # leak / capacitance / vleak
#         if self._fix_vleak is None:
#             self.vleak = nn.Parameter(
#                 torch.empty(self._num_units, device=self.device, dtype=self.dtype).uniform_(-0.2, 0.2)
#             )
#         else:
#             # fixed vleak stored as buffer (not trainable)
#             self.register_buffer("vleak", torch.tensor(self._fix_vleak, device=self.device, dtype=self.dtype))

#         if self._fix_gleak is None:
#             if self._gleak_init_max > self._gleak_init_min:
#                 gleak_val = torch.empty(self._num_units, device=self.device, dtype=self.dtype).uniform_(
#                     self._gleak_init_min, self._gleak_init_max
#                 )
#             else:
#                 gleak_val = torch.full((self._num_units,), fill_value=self._gleak_init_min, device=self.device, dtype=self.dtype)
#             self.gleak = nn.Parameter(gleak_val)
#         else:
#             self.register_buffer("gleak", torch.tensor(self._fix_gleak, device=self.device, dtype=self.dtype))

#         if self._fix_cm is None:
#             if self._cm_init_max > self._cm_init_min:
#                 cm_val = torch.empty(self._num_units, device=self.device, dtype=self.dtype).uniform_(
#                     self._cm_init_min, self._cm_init_max
#                 )
#             else:
#                 cm_val = torch.full((self._num_units,), fill_value=self._cm_init_min, device=self.device, dtype=self.dtype)
#             self.cm_t = nn.Parameter(cm_val)
#         else:
#             self.register_buffer("cm_t", torch.tensor(self._fix_cm, device=self.device, dtype=self.dtype))

#         # Input mapping parameters (for Linear/Affine)
#         if self._input_mapping in (MappingType.Linear, MappingType.Affine):
#             # weights per input feature
#             self.input_w = nn.Parameter(torch.ones(self._input_size, device=self.device, dtype=self.dtype))
#         if self._input_mapping is MappingType.Affine:
#             self.input_b = nn.Parameter(torch.zeros(self._input_size, device=self.device, dtype=self.dtype))

#         self._params_created = True

#     def forward(self, inputs, state):
#         """
#         inputs: (batch, input_size)
#         state: (batch, num_units)
#         returns: (outputs, next_state) where both are (batch, num_units)
#         """
#         # lazy build
#         if not self._is_built:
#             self._is_built = True
#             self._input_size = int(inputs.shape[-1])
#             self._get_variables()
#         else:
#             if self._input_size != int(inputs.shape[-1]):
#                 raise ValueError(
#                     "You first feed an input with {} features and now one with {} features, that is not possible".format(
#                         self._input_size, int(inputs.shape[-1])
#                     )
#                 )

#         inputs = self._map_inputs(inputs)

#         if self._solver == ODESolver.Explicit:
#             next_state = self._ode_step_explicit(inputs, state, _ode_solver_unfolds=self._ode_solver_unfolds)
#         elif self._solver == ODESolver.SemiImplicit:
#             next_state = self._ode_step(inputs, state)
#         elif self._solver == ODESolver.RungeKutta:
#             next_state = self._ode_step_runge_kutta(inputs, state)
#         else:
#             raise ValueError("Unknown ODE solver '{}'".format(str(self._solver)))

#         outputs = next_state
#         return outputs, next_state

#     def _sigmoid(self, v_pre, mu, sigma):
#         """
#         v_pre: (batch, pre)  or (batch, input_size) depending on call
#         mu: (pre, post)
#         sigma: (pre, post)
#         returns: (batch, pre, post)
#         """
#         # reshape v_pre to (batch, pre, 1)
#         v_pre_reshaped = v_pre.unsqueeze(-1)  # (batch, pre, 1)
#         # broadcasting: (batch, pre, 1) - (pre, post) -> (batch, pre, post)
#         mues = v_pre_reshaped - mu  # broadcasting
#         x = sigma * mues
#         return torch.sigmoid(x)

#     def _ode_step(self, inputs, state):
#         """
#         Semi-implicit / hybrid Euler method (matches TF code)
#         inputs: (batch, input_size)
#         state: (batch, num_units)
#         """
#         v_pre = state

#         # sensory part
#         sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)  # (batch, input_size, num_units)
#         sensory_rev_activation = sensory_w_activation * self.sensory_erev  # (batch, input_size, num_units)

#         w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)  # (batch, num_units)
#         w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)  # (batch, num_units)

#         for t in range(self._ode_solver_unfolds):
#             w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)  # (batch, num_units, num_units)
#             rev_activation = w_activation * self.erev  # (batch, num_units, num_units)

#             w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory  # (batch, num_units)
#             w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory  # (batch, num_units)

#             # numerator/denominator shapes align by broadcasting
#             numerator = self.cm_t.unsqueeze(0) * v_pre + self.gleak.unsqueeze(0) * self.vleak.unsqueeze(0) + w_numerator
#             denominator = self.cm_t.unsqueeze(0) + self.gleak.unsqueeze(0) + w_denominator

#             v_pre = numerator / denominator

#         return v_pre

#     def _f_prime(self, inputs, state):
#         """
#         Derivative helper used by Runge-Kutta.
#         """
#         v_pre = state

#         sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)  # (batch, in, out)
#         w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)  # (batch, out)

#         w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)  # (batch, pre, post)
#         w_reduced_synapse = torch.sum(w_activation, dim=1)  # (batch, post)

#         sensory_in = self.sensory_erev * sensory_w_activation  # (batch, in, out)
#         synapse_in = self.erev * w_activation  # (batch, pre, post)

#         sum_in = torch.sum(sensory_in, dim=1) - v_pre * w_reduced_synapse + torch.sum(synapse_in, dim=1) - v_pre * w_reduced_sensory

#         f_prime = (1.0 / self.cm_t.unsqueeze(0)) * (self.gleak.unsqueeze(0) * (self.vleak.unsqueeze(0) - v_pre) + sum_in)
#         return f_prime

#     def _ode_step_runge_kutta(self, inputs, state):
#         h = 0.1
#         for i in range(self._ode_solver_unfolds):
#             k1 = h * self._f_prime(inputs, state)
#             k2 = h * self._f_prime(inputs, state + k1 * 0.5)
#             k3 = h * self._f_prime(inputs, state + k2 * 0.5)
#             k4 = h * self._f_prime(inputs, state + k3)
#             state = state + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
#         return state

#     def _ode_step_explicit(self, inputs, state, _ode_solver_unfolds=None):
#         if _ode_solver_unfolds is None:
#             _ode_solver_unfolds = self._ode_solver_unfolds

#         v_pre = state

#         sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
#         w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)  # (batch, num_units)

#         for t in range(_ode_solver_unfolds):
#             w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)  # (batch, pre, post)
#             w_reduced_synapse = torch.sum(w_activation, dim=1)

#             sensory_in = self.sensory_erev * sensory_w_activation
#             synapse_in = self.erev * w_activation

#             sum_in = torch.sum(sensory_in, dim=1) - v_pre * w_reduced_synapse + torch.sum(synapse_in, dim=1) - v_pre * w_reduced_sensory

#             f_prime = (1.0 / self.cm_t.unsqueeze(0)) * (self.gleak.unsqueeze(0) * (self.vleak.unsqueeze(0) - v_pre) + sum_in)

#             v_pre = v_pre + 0.1 * f_prime

#         return v_pre

#     def get_param_constrain_op(self):
#         """
#         Clamp parameters in-place (similar to TF assign with clip_by_value).
#         """
#         # Note: use .data.clamp_ to mutate parameters in-place outside grad tracking.
#         if hasattr(self, "cm_t"):
#             if isinstance(self.cm_t, nn.Parameter):
#                 self.cm_t.data.clamp_(self._cm_t_min_value, self._cm_t_max_value)
#         if hasattr(self, "gleak"):
#             if isinstance(self.gleak, nn.Parameter):
#                 self.gleak.data.clamp_(self._gleak_min_value, self._gleak_max_value)
#         if hasattr(self, "W"):
#             self.W.data.clamp_(self._w_min_value, self._w_max_value)
#         if hasattr(self, "sensory_W"):
#             self.sensory_W.data.clamp_(self._w_min_value, self._w_max_value)

#         # nothing to return; params are clamped in-place

#     def export_weights(self, dirname, output_weights=None):
#         """
#         Save weights to CSV files (numpy text) like the TF implementation.
#         output_weights: tuple (output_w_tensor, output_b_tensor) that will be saved if provided.
#         """
#         os.makedirs(dirname, exist_ok=True)
#         with torch.no_grad():
#             w = self.W.cpu().numpy()
#             erev = self.erev.cpu().numpy()
#             mu = self.mu.cpu().numpy()
#             sigma = self.sigma.cpu().numpy()
#             sensory_w = self.sensory_W.cpu().numpy()
#             sensory_erev = self.sensory_erev.cpu().numpy()
#             sensory_mu = self.sensory_mu.cpu().numpy()
#             sensory_sigma = self.sensory_sigma.cpu().numpy()
#             vleak = self.vleak.cpu().numpy()
#             gleak = self.gleak.cpu().numpy()
#             cm = self.cm_t.cpu().numpy()

#             np.savetxt(os.path.join(dirname, "w.csv"), w)
#             np.savetxt(os.path.join(dirname, "erev.csv"), erev)
#             np.savetxt(os.path.join(dirname, "mu.csv"), mu)
#             np.savetxt(os.path.join(dirname, "sigma.csv"), sigma)
#             np.savetxt(os.path.join(dirname, "sensory_w.csv"), sensory_w)
#             np.savetxt(os.path.join(dirname, "sensory_erev.csv"), sensory_erev)
#             np.savetxt(os.path.join(dirname, "sensory_mu.csv"), sensory_mu)
#             np.savetxt(os.path.join(dirname, "sensory_sigma.csv"), sensory_sigma)
#             np.savetxt(os.path.join(dirname, "vleak.csv"), vleak)
#             np.savetxt(os.path.join(dirname, "gleak.csv"), gleak)
#             np.savetxt(os.path.join(dirname, "cm.csv"), cm)

#             if output_weights is not None:
#                 output_w, output_b = output_weights
#                 np.savetxt(os.path.join(dirname, "output_w.csv"), output_w.detach().cpu().numpy())
#                 np.savetxt(os.path.join(dirname, "output_b.csv"), output_b.detach().cpu().numpy())
