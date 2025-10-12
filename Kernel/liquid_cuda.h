#pragma once
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> liquid_forward(
    torch::Tensor x, torch::Tensor h_init, torch::Tensor W_in, torch::Tensor W_rec,
    torch::Tensor bias, torch::Tensor tau_param, float dt);

std::vector<torch::Tensor> liquid_backward(
    torch::Tensor grad_out, torch::Tensor x, torch::Tensor h_init, torch::Tensor W_in,
    torch::Tensor W_rec, torch::Tensor tau_param, torch::Tensor outputs,
    torch::Tensor pre_activations, float dt);

