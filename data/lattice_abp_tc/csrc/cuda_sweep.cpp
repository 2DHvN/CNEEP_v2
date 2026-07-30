#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

void launch_lattice_abp_cuda_sweep(
    at::Tensor sites,
    at::Tensor occupancy,
    const at::Tensor& order,
    const at::Tensor& draws,
    const at::Tensor& active_work,
    const at::Tensor& kernel_offsets,
    const at::Tensor& kernel_values,
    const at::Tensor& neighbor_linear,
    double base,
    double reservoir_diffusion,
    double mobility,
    int64_t prefactor_code,
    bool strict_probabilities,
    double probability_tolerance,
    bool return_ep_map,
    at::Tensor probabilities,
    at::Tensor delta_potential,
    at::Tensor total_ep,
    at::Tensor active_ep,
    at::Tensor wca_ep,
    at::Tensor accepted_hops,
    at::Tensor ep_map,
    at::Tensor status,
    at::Tensor failed_order_index,
    at::Tensor bad_max_sum);

namespace {

void check_cuda_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void check_same_device(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name) {
  TORCH_CHECK(
      tensor.get_device() == reference.get_device(),
      name,
      " must be on the same CUDA device as sites.");
}

}  // namespace

std::vector<at::Tensor> sweep_cuda(
    at::Tensor sites,
    at::Tensor occupancy,
    const at::Tensor& order,
    const at::Tensor& draws,
    const at::Tensor& active_work,
    const at::Tensor& kernel_offsets,
    const at::Tensor& kernel_values,
    const at::Tensor& neighbor_linear,
    double base,
    double reservoir_diffusion,
    double mobility,
    int64_t prefactor_code,
    bool strict_probabilities,
    double probability_tolerance,
    bool return_ep_map) {
  check_cuda_contiguous(sites, "sites");
  check_cuda_contiguous(occupancy, "occupancy");
  check_cuda_contiguous(order, "order");
  check_cuda_contiguous(draws, "draws");
  check_cuda_contiguous(active_work, "active_work");
  check_cuda_contiguous(kernel_offsets, "kernel_offsets");
  check_cuda_contiguous(kernel_values, "kernel_values");
  check_cuda_contiguous(neighbor_linear, "neighbor_linear");

  check_same_device(occupancy, sites, "occupancy");
  check_same_device(order, sites, "order");
  check_same_device(draws, sites, "draws");
  check_same_device(active_work, sites, "active_work");
  check_same_device(kernel_offsets, sites, "kernel_offsets");
  check_same_device(kernel_values, sites, "kernel_values");
  check_same_device(neighbor_linear, sites, "neighbor_linear");

  TORCH_CHECK(
      sites.scalar_type() == at::kLong && sites.dim() == 3 &&
          sites.size(2) == 2,
      "sites must be contiguous int64 with shape [B, N, 2].");
  TORCH_CHECK(
      occupancy.scalar_type() == at::kLong && occupancy.dim() == 3 &&
          occupancy.size(1) == occupancy.size(2),
      "occupancy must be contiguous int64 with shape [B, G, G].");

  const int64_t batch_size = sites.size(0);
  const int64_t particle_count = sites.size(1);
  const int64_t grid_size = occupancy.size(1);
  TORCH_CHECK(batch_size > 0, "The CUDA sweep requires B > 0.");
  TORCH_CHECK(
      static_cast<uint64_t>(batch_size) <=
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
      "The CUDA sweep batch dimension is too large for a CUDA grid.");
  TORCH_CHECK(particle_count > 0, "The CUDA sweep requires N > 0.");
  TORCH_CHECK(grid_size > 0, "The CUDA sweep requires G > 0.");
  TORCH_CHECK(
      occupancy.size(0) == batch_size,
      "sites and occupancy batch dimensions must match.");
  TORCH_CHECK(
      order.scalar_type() == at::kLong && order.dim() == 1 &&
          order.size(0) == particle_count,
      "order must be int64 with shape [N].");
  TORCH_CHECK(
      draws.dim() == 2 && draws.size(0) == particle_count &&
          draws.size(1) == batch_size,
      "draws must have shape [N, B].");
  TORCH_CHECK(
      active_work.dim() == 3 && active_work.size(0) == batch_size &&
          active_work.size(1) == particle_count && active_work.size(2) == 4,
      "active_work must have shape [B, N, 4].");
  TORCH_CHECK(
      kernel_offsets.scalar_type() == at::kLong &&
          kernel_offsets.dim() == 2 && kernel_offsets.size(1) == 2,
      "kernel_offsets must be int64 with shape [K, 2].");
  TORCH_CHECK(
      kernel_values.dim() == 1 &&
          kernel_values.size(0) == kernel_offsets.size(0),
      "kernel_values must have shape [K].");
  TORCH_CHECK(
      kernel_values.scalar_type() == at::kFloat ||
          kernel_values.scalar_type() == at::kDouble,
      "kernel_values must have dtype float32 or float64.");
  TORCH_CHECK(
      draws.scalar_type() == kernel_values.scalar_type(),
      "draws and kernel_values must have the same dtype.");
  TORCH_CHECK(
      active_work.scalar_type() == kernel_values.scalar_type(),
      "active_work and kernel_values must have the same dtype.");
  TORCH_CHECK(
      neighbor_linear.numel() == 0 ||
          (neighbor_linear.scalar_type() == at::kInt &&
           neighbor_linear.dim() == 2 &&
           neighbor_linear.size(0) == grid_size * grid_size &&
           neighbor_linear.size(1) == kernel_offsets.size(0)),
      "neighbor_linear must be empty or int32 with shape [G * G, K].");
  TORCH_CHECK(base > 0.0, "base must be positive.");
  TORCH_CHECK(
      reservoir_diffusion > 0.0,
      "reservoir_diffusion must be positive.");
  TORCH_CHECK(mobility > 0.0, "mobility must be positive.");
  TORCH_CHECK(
      prefactor_code == 0 || prefactor_code == 1,
      "prefactor_code must be 0 (cv) or 1 (c0).");
  TORCH_CHECK(
      probability_tolerance >= 0.0,
      "probability_tolerance must be nonnegative.");

  c10::cuda::CUDAGuard device_guard(sites.device());
  const auto float_options = kernel_values.options();
  const auto long_options = sites.options().dtype(at::kLong);
  const auto int_options = sites.options().dtype(at::kInt);

  auto probabilities = at::empty({batch_size, 4}, float_options);
  auto delta_potential = at::empty({batch_size, 4}, float_options);
  auto total_ep = at::zeros({batch_size}, float_options);
  auto active_ep = at::zeros({batch_size}, float_options);
  auto wca_ep = at::zeros({batch_size}, float_options);
  auto accepted_hops = at::zeros({batch_size}, long_options);
  auto ep_map = return_ep_map
      ? at::zeros({batch_size, grid_size, grid_size}, float_options)
      : at::empty({0}, float_options);
  auto status = at::zeros({}, int_options);
  auto failed_order_index = at::full({}, -1, long_options);
  auto bad_max_sum = at::zeros({}, float_options);

  launch_lattice_abp_cuda_sweep(
      sites,
      occupancy,
      order,
      draws,
      active_work,
      kernel_offsets,
      kernel_values,
      neighbor_linear,
      base,
      reservoir_diffusion,
      mobility,
      prefactor_code,
      strict_probabilities,
      probability_tolerance,
      return_ep_map,
      probabilities,
      delta_potential,
      total_ep,
      active_ep,
      wca_ep,
      accepted_hops,
      ep_map,
      status,
      failed_order_index,
      bad_max_sum);

  return {
      total_ep,
      active_ep,
      wca_ep,
      accepted_hops,
      ep_map,
      status,
      failed_order_index,
      bad_max_sum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "sweep_cuda",
      &sweep_cuda,
      "Fused exact random-sequential lattice-ABP CUDA sweep");
}
