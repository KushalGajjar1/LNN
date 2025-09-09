import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import os
from enum import Enum

class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

class LTCCell(nn.Module):

    def __init__(self, input_size, num_units, solver, solver_clip=5.0):
        super(LTCCell, self).__init__()

        self._input_size = input_size
        self._num_units = num_units
        self._solver = solver
        self._solver_clip = solver_clip
        
        self._ode_solver_unfolds = 6
        self._input_mapping = MappingType.Affine
        self._erev_init_factor = 1
        self._w_init_max = 1.0
        self._w_init_min = 0.01
        self._cm_init_min = 0.5
        self._cm_init_max = 0.5
        self._gleak_init_min = 1.0
        self._gleak_init_max = 1.0
        self._fix_cm = None
        self._fix_gleak = None
        self._fix_vleak = None

        self._w_min_value = 1e-5
        self._w_max_value = 1000.0
        self._gleak_min_value = 1e-5
        self._gleak_max_value = 1000.0
        self._cm_t_min_value = 1e-6
        self._cm_t_max_value = 1000.0

        self._get_variables()
        self._map_inputs()

    @property
    def state_size(self):
        return self._num_units
    
    @property
    def output_size(self):
        return self._num_units
    
    def _map_inputs(self):

        if self._input_mapping in [MappingType.Affine, MappingType.Linear]:
            self.input_w = nn.Parameter(torch.Tensor(self._input_size))
            init.constant_(self.input_w, 1.0)
        if self._input_mapping == MappingType.Affine:
            self.input_b = nn.Parameter(torch.Tensor(self._input_size))
            init.constant_(self.input_b, 0.0)

    def _get_variables(self):
        
        self.sensory_mu = nn.Parameter(torch.Tensor(self._input_size, self._num_units))
        self.sensory_sigma = nn.Parameter(torch.Tensor(self._input_size, self._num_units))
        self.sensory_W = nn.Parameter(torch.Tensor(self._input_size, self._num_units))
        sensory_erev_init = (2 * np.random.randint(0, 2, size=[self._input_size, self._num_units]) - 1) * self._erev_init_factor
        self.sensory_erev = nn.Parameter(torch.from_numpy(sensory_erev_init).float())

        init.uniform_(self.sensory_mu, a=0.3, b=0.8)
        init.uniform_(self.sensory_sigma, a=3.0, b=8.0)
        init.uniform_(self.sensory_W, a=self._w_init_min, b=self._w_init_max)

        self.mu = nn.Parameter(torch.Tensor(self._num_units, self._num_units))
        self.sigma = nn.Parameter(torch.Tensor(self._num_units, self._num_units))
        self.W = nn.Parameter(torch.Tensor(self._num_units, self._num_units))
        erev_init = (2 * np.random.randint(0, 2, size=[self._num_units, self._num_units]) - 1) * self._erev_init_factor
        self.erev = nn.Parameter(torch.from_numpy(erev_init).float())

        init.uniform_(self.mu, a=0.3, b=0.8)
        init.uniform_(self.sigma, a=3.0, b=8.0)
        init.uniform_(self.W, a=self._w_init_min, b=self._w_init_max)

        if self._fix_vleak is None:
            self.vleak = nn.Parameter(torch.Tensor(self._num_units))
            init.uniform_(self.vleak, a=-0.2, b=0.2)
        else:
            self.register_buffer('vleak', torch.full([self._num_units], self._fix_vleak))

        if self._fix_gleak is None:
            self.gleak = nn.Parameter(torch.Tensor(self._num_units))
            if self._gleak_init_max > self._gleak_init_min:
                init.uniform_(self.gleak, a=self._gleak_init_min, b=self._gleak_init_max)
            else:
                init.constant_(self.gleak, self._gleak_init_min)
        else:
            self.register_buffer('gleak', torch.full([self._num_units], self._fix_gleak))

        if self._fix_cm is None:
            self.cm_t = nn.Parameter(torch.Tensor(self._num_units))
            if self._cm_init_max > self._cm_init_min:
                init.uniform_(self.cm_t, a=self._cm_init_min, b=self._cm_init_max)
            else:
                init.constant_(self.cm_t, self._cm_init_min)
        else:
            self.register_buffer('cm_t', torch.full([self._num_units], self._fix_cm))

    def forward(self, inputs, state):

        if self._input_mapping in [MappingType.Affine, MappingType.Linear]:
            inputs = inputs * self.input_w
        if self._input_mapping == MappingType.Affine:
            inputs = inputs + self.input_b
        
        if self._solver == ODESolver.Explicit:
            next_state = self._ode_step_explicit(inputs, state)
        elif self._solver == ODESolver.SemiImplicit:
            next_state = self._ode_step_semi_implicit(inputs, state)
        elif self._solver == ODESolver.RungeKutta:
            next_state = self._ode_step_runge_kutta(inputs, state)
        else:
            raise ValueError(f"Unknown ODE solver '{str(self._solver)}'")
        
        return next_state, next_state

    def _ode_step_semi_implicit(self, inputs, state):
        v_pre = state
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)
        for _ in range(self._ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory
            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator
            v_pre = numerator / denominator
        return v_pre

    def _f_prime(self, inputs, state):
        v_pre = state
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)
        w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
        w_reduced_synapse = torch.sum(w_activation, dim=1)
        sensory_in = self.sensory_erev * sensory_w_activation
        synapse_in = self.erev * w_activation
        sum_in = (torch.sum(sensory_in, dim=1) - v_pre * w_reduced_synapse +
                  torch.sum(synapse_in, dim=1) - v_pre * w_reduced_sensory)
        f_prime = (1 / self.cm_t) * (self.gleak * (self.vleak - v_pre) + sum_in)
        return f_prime
    
    def _ode_step_explicit(self, inputs, state):
        v_pre = state
        h = 0.1
        for _ in range(self._ode_solver_unfolds):
            f_prime = self._f_prime(inputs, v_pre)
            v_pre = v_pre + h * f_prime
            if self._solver_clip > 0:
                v_pre = torch.clamp(v_pre, -self._solver_clip, self._solver_clip)
        return v_pre
    
    def _ode_step_runge_kutta(self, inputs, state):
        v_pre = state
        h = 0.1
        for _ in range(self._ode_solver_unfolds):
            k1 = h * self._f_prime(inputs, v_pre)
            k2 = h * self._f_prime(inputs, v_pre + 0.5 * k1)
            k3 = h * self._f_prime(inputs, v_pre + 0.5 * k2)
            k4 = h * self._f_prime(inputs, v_pre + k3)
            v_pre = v_pre + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if self._solver_clip > 0:
                v_pre = torch.clamp(v_pre, -self._solver_clip, self._solver_clip)
        return v_pre
    
    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = v_pre.unsqueeze(-1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)
    
    def constrain_parameters(self):
        self.cm_t.data.clamp_(min=self._cm_t_min_value, max=self._cm_t_max_value)
        self.gleak.data.clamp_(min=self._gleak_min_value, max=self._gleak_max_value)
        self.W.data.clamp_(min=self._w_min_value, max=self._w_max_value)
        self.sensory_W.data.clamp_(min=self._w_min_value, max=self._w_max_value)
