// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <torch/nativert/executor/triton/TritonKernelManager.h>

#include <ATen/Functions.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include <triton_mtia/python/mtia/sigmoid/sigmoid_launcher.h>

namespace torch::nativert {

class MtiaKernelInputs : public KernelInputs {
 public:
  MtiaKernelInputs(
      size_t num_args,
      size_t num_attrs,
      const KernelInputParams& params)
      : KernelInputs(num_args, num_attrs),
      input_manager_(num_args, num_attrs, params.kernel_param_types) {
  }

  void add_arg(void *arg) override {
    TORCH_CHECK(false, "add_arg not supported for MTIA!");
  }

  void add_tensor_arg(const at::Tensor& tensor) override {
    input_manager_.add_tensor_arg(tensor);
  }

  void add_attribute(void* attr) override {
    input_manager_.add_attribute(attr);
  }

  void** as_void() override {
    return input_manager_.as_void();
  }

 private:
  mtia::sigmoid::KernelInputManager input_manager_;
};

class MtiaTritonKernelManager final : public TritonKernelManager {
 public:
  MtiaTritonKernelManager(
      std::string kernel_name,
      std::string kernel_bin_path,
      std::string kernel_launcher_bin_path);
  ~MtiaTritonKernelManager() final = default;

  std::unique_ptr<KernelInputs> create_inputs(
      size_t num_args,
      size_t num_attrs,
      const KernelInputParams& params) const override {
    // Store params for use in ensureLoaded during launch
    kernel_input_params_ = params;
    launcher_ = std::make_unique<mtia::sigmoid::KernelLauncher>(
      kernel_name_, kernel_bin_path_, params.kernel_param_names,
      params.kernel_param_types, params.output_indices);
    return std::make_unique<MtiaKernelInputs>(num_args, num_attrs, params);
  }

  void launch(const LaunchParams& launch_params, void** args) override;

 private:
  mutable KernelInputParams kernel_input_params_;
  mutable std::unique_ptr<mtia::sigmoid::KernelLauncher> launcher_;
};

MtiaTritonKernelManager::MtiaTritonKernelManager(
    std::string kernel_name,
    std::string kernel_bin_path,
    std::string kernel_launcher_bin_path)
    : TritonKernelManager(std::move(kernel_name), std::move(kernel_bin_path)),
    launcher_(nullptr) {}


void MtiaTritonKernelManager::launch(
    const LaunchParams& launch_params,
    void** args) {
  TORCH_CHECK(launcher_ != nullptr, "Kernel not loaded, create_inputs must be called before launching!");
  launcher_->launch(launch_params.grid_dims.x,
    launch_params.grid_dims.y,
    launch_params.grid_dims.z,
    launch_params.mtia_tile_width,
    launch_params.mtia_tile_height,
    launch_params.mtia_base_pe,
    kernel_input_params_.kernel_param_names.size(),
    args);
}

namespace {
std::unique_ptr<TritonKernelManager> create_mtia_triton_kernel_manager(
    std::string kernel_name,
    std::string kernel_bin_path,
    std::string kernel_launcher_bin_path) {
  return std::make_unique<MtiaTritonKernelManager>(
      std::move(kernel_name),
      std::move(kernel_bin_path),
      std::move(kernel_launcher_bin_path));
}
} // namespace

C10_REGISTER_TYPED_CREATOR(
    TritonKernelManagerRegistry,
    at::kMTIA,
    create_mtia_triton_kernel_manager)

} // namespace torch::nativert
