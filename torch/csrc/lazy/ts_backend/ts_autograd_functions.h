#pragma once

#include <torch/csrc/autograd/custom_function.h>
#include <torch/torch.h>

namespace torch {
namespace lazy {

struct MaxPool3dAutogradFunctionTS
    : public torch::autograd::Function<MaxPool3dAutogradFunctionTS> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor self,
                               torch::IntArrayRef kernel_size,
                               torch::IntArrayRef stride,
                               torch::IntArrayRef padding,
                               torch::IntArrayRef dilation, bool ceil_mode);
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output);
};

}  // namespace lazy
}  // namespace torch
