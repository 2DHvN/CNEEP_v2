#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kDirectionCount = 4;
constexpr int kQueryCount = 5;
constexpr int kWarpSize = 32;
constexpr int kThreadCount = kQueryCount * kWarpSize;
constexpr unsigned int kFullWarpMask = 0xffffffffu;

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_exp(scalar_t value);

template <>
__device__ __forceinline__ float device_exp<float>(float value) {
  return expf(value);
}

template <>
__device__ __forceinline__ double device_exp<double>(double value) {
  return exp(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_expm1(scalar_t value);

template <>
__device__ __forceinline__ float device_expm1<float>(float value) {
  return expm1f(value);
}

template <>
__device__ __forceinline__ double device_expm1<double>(double value) {
  return expm1(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_abs(scalar_t value);

template <>
__device__ __forceinline__ float device_abs<float>(float value) {
  return fabsf(value);
}

template <>
__device__ __forceinline__ double device_abs<double>(double value) {
  return fabs(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t positive_infinity();

template <>
__device__ __forceinline__ float positive_infinity<float>() {
  return CUDART_INF_F;
}

template <>
__device__ __forceinline__ double positive_infinity<double>() {
  return CUDART_INF;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t dtype_epsilon();

template <>
__device__ __forceinline__ float dtype_epsilon<float>() {
  return FLT_EPSILON;
}

template <>
__device__ __forceinline__ double dtype_epsilon<double>() {
  return DBL_EPSILON;
}

__device__ __forceinline__ int64_t periodic_index(
    int64_t value,
    int64_t grid_size) {
  value %= grid_size;
  return value < 0 ? value + grid_size : value;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_sum(scalar_t value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t cv_factor(scalar_t x) {
  const scalar_t zero = static_cast<scalar_t>(0);
  const scalar_t one = static_cast<scalar_t>(1);
  const scalar_t two = static_cast<scalar_t>(2);
  scalar_t value;
  if (device_abs(x) < static_cast<scalar_t>(1.0e-5)) {
    value = one + x + (x * x) / static_cast<scalar_t>(3);
  } else if (x > static_cast<scalar_t>(50)) {
    value = two * x;
  } else if (x < static_cast<scalar_t>(-50)) {
    value = (-two * x) * device_exp(two * x);
  } else {
    value = (two * x) / (-device_expm1(-two * x));
  }
  return value < zero ? zero : value;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t transition_probability(
    scalar_t active,
    scalar_t energy_change,
    scalar_t base,
    scalar_t reservoir_diffusion,
    scalar_t mobility,
    int prefactor_code) {
  const scalar_t zero = static_cast<scalar_t>(0);
  const scalar_t two = static_cast<scalar_t>(2);
  const scalar_t affinity = active - mobility * energy_change;
  const scalar_t x = affinity / (two * reservoir_diffusion);
  if (!isfinite(x) || !isfinite(energy_change)) {
    return zero;
  }
  scalar_t probability = prefactor_code == 1
      ? base * device_exp(x)
      : base * cv_factor(x);
  return probability < zero ? zero : probability;
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreadCount) void lattice_abp_sweep_global_kernel(
    int64_t* __restrict__ sites,
    int64_t* __restrict__ occupancy,
    const int64_t* __restrict__ order,
    const scalar_t* __restrict__ draws,
    const scalar_t* __restrict__ active_work,
    const int64_t* __restrict__ kernel_offsets,
    const scalar_t* __restrict__ kernel_values,
    const int32_t* __restrict__ neighbor_linear,
    int64_t batch_size,
    int64_t particle_count,
    int64_t grid_size,
    int64_t kernel_size,
    scalar_t base,
    scalar_t reservoir_diffusion,
    scalar_t mobility,
    int prefactor_code,
    bool strict_probabilities,
    scalar_t probability_tolerance,
    bool record_ep_map,
    scalar_t* __restrict__ probabilities,
    scalar_t* __restrict__ delta_potential,
    scalar_t* __restrict__ total_ep,
    scalar_t* __restrict__ active_ep,
    scalar_t* __restrict__ wca_ep,
    int64_t* __restrict__ accepted_hops,
    scalar_t* __restrict__ ep_map,
    int32_t* __restrict__ status,
    int64_t* __restrict__ failed_order_index,
    scalar_t* __restrict__ bad_max_sum) {
  __shared__ scalar_t query_energies[kQueryCount];
  __shared__ int64_t query_x[kQueryCount];
  __shared__ int64_t query_y[kQueryCount];
  __shared__ int destination_occupied[kDirectionCount];
  __shared__ int abort_sweep;

  const int thread_index = static_cast<int>(threadIdx.x);
  const int query_index = thread_index / kWarpSize;
  const int lane_index = thread_index % kWarpSize;

  for (int64_t order_index = 0; order_index < particle_count; ++order_index) {
    const int64_t particle_index = order[order_index];

    // All ensembles remove the current particle before any local energy is
    // evaluated, exactly as in the reference random-sequential path.
    for (int64_t batch_index = thread_index; batch_index < batch_size;
         batch_index += blockDim.x) {
      const int64_t site_offset =
          (batch_index * particle_count + particle_index) * 2;
      const int64_t old_x = sites[site_offset];
      const int64_t old_y = sites[site_offset + 1];
      const int64_t occupancy_offset =
          (batch_index * grid_size + old_x) * grid_size + old_y;
      occupancy[occupancy_offset] -= 1;
    }
    __syncthreads();

    // Five warps cooperatively compute old,+x,-x,+y,-y energies.  Batches are
    // deliberately processed together inside one block so a strict failure
    // can be detected across all B before any batch commits its hop.
    for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      if (thread_index == 0) {
        const int64_t site_offset =
            (batch_index * particle_count + particle_index) * 2;
        const int64_t old_x = sites[site_offset];
        const int64_t old_y = sites[site_offset + 1];
        query_x[0] = old_x;
        query_y[0] = old_y;
        query_x[1] = periodic_index(old_x + 1, grid_size);
        query_y[1] = old_y;
        query_x[2] = periodic_index(old_x - 1, grid_size);
        query_y[2] = old_y;
        query_x[3] = old_x;
        query_y[3] = periodic_index(old_y + 1, grid_size);
        query_x[4] = old_x;
        query_y[4] = periodic_index(old_y - 1, grid_size);
        for (int direction = 0; direction < kDirectionCount; ++direction) {
          const int query = direction + 1;
          const int64_t occupancy_offset =
              (batch_index * grid_size + query_x[query]) * grid_size +
              query_y[query];
          destination_occupied[direction] =
              occupancy[occupancy_offset] > 0 ? 1 : 0;
        }
      }
      __syncthreads();

      scalar_t energy = static_cast<scalar_t>(0);
      const bool skip_query =
          query_index > 0 && destination_occupied[query_index - 1] != 0;
      if (!skip_query) {
        const int64_t center_x = query_x[query_index];
        const int64_t center_y = query_y[query_index];
        const int64_t center_linear = center_x * grid_size + center_y;
        for (int64_t kernel_index = lane_index; kernel_index < kernel_size;
             kernel_index += kWarpSize) {
          int64_t neighbor_site;
          if (neighbor_linear != nullptr) {
            neighbor_site = static_cast<int64_t>(
                neighbor_linear[center_linear * kernel_size + kernel_index]);
          } else {
            const int64_t neighbor_x = periodic_index(
                center_x + kernel_offsets[kernel_index * 2],
                grid_size);
            const int64_t neighbor_y = periodic_index(
                center_y + kernel_offsets[kernel_index * 2 + 1],
                grid_size);
            neighbor_site = neighbor_x * grid_size + neighbor_y;
          }
          const int64_t occupancy_offset =
              batch_index * grid_size * grid_size + neighbor_site;
          energy += static_cast<scalar_t>(occupancy[occupancy_offset]) *
              kernel_values[kernel_index];
        }
      }
      energy = warp_sum(energy);
      if (lane_index == 0) {
        query_energies[query_index] = energy;
      }
      __syncthreads();

      if (thread_index == 0) {
        for (int direction = 0; direction < kDirectionCount; ++direction) {
          const int probability_offset =
              static_cast<int>(batch_index * kDirectionCount + direction);
          const scalar_t energy_change = destination_occupied[direction]
              ? positive_infinity<scalar_t>()
              : query_energies[direction + 1] - query_energies[0];
          delta_potential[probability_offset] = energy_change;
          const int64_t active_offset =
              (batch_index * particle_count + particle_index) *
                  kDirectionCount +
              direction;
          probabilities[probability_offset] = transition_probability(
              active_work[active_offset],
              energy_change,
              base,
              reservoir_diffusion,
              mobility,
              prefactor_code);
        }
      }
      __syncthreads();
    }

    if (thread_index == 0) {
      abort_sweep = 0;
      bool any_nonfinite = false;
      for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        for (int direction = 0; direction < kDirectionCount; ++direction) {
          const int64_t offset =
              batch_index * kDirectionCount + direction;
          if (!isfinite(probabilities[offset])) {
            any_nonfinite = true;
          }
        }
      }

      if (any_nonfinite && strict_probabilities) {
        *status = 1;
        *failed_order_index = order_index;
        abort_sweep = 1;
      } else {
        if (any_nonfinite) {
          for (int64_t batch_index = 0; batch_index < batch_size;
               ++batch_index) {
            for (int direction = 0; direction < kDirectionCount; ++direction) {
              const int64_t offset =
                  batch_index * kDirectionCount + direction;
              const scalar_t value = probabilities[offset];
              if (isnan(value) || value < static_cast<scalar_t>(0)) {
                probabilities[offset] = static_cast<scalar_t>(0);
              } else if (isinf(value)) {
                probabilities[offset] = static_cast<scalar_t>(1);
              }
            }
          }
        }

        scalar_t maximum_sum = static_cast<scalar_t>(0);
        for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
          scalar_t probability_sum = static_cast<scalar_t>(0);
          for (int direction = 0; direction < kDirectionCount; ++direction) {
            probability_sum +=
                probabilities[batch_index * kDirectionCount + direction];
          }
          if (batch_index == 0 || probability_sum > maximum_sum) {
            maximum_sum = probability_sum;
          }
        }

        const scalar_t threshold =
            static_cast<scalar_t>(1) + probability_tolerance;
        if (maximum_sum > threshold && strict_probabilities) {
          *status = 2;
          *failed_order_index = order_index;
          *bad_max_sum = maximum_sum;
          abort_sweep = 1;
        } else if (maximum_sum > threshold) {
          for (int64_t batch_index = 0; batch_index < batch_size;
               ++batch_index) {
            scalar_t probability_sum = static_cast<scalar_t>(0);
            for (int direction = 0; direction < kDirectionCount; ++direction) {
              probability_sum +=
                  probabilities[batch_index * kDirectionCount + direction];
            }
            const scalar_t denominator =
                probability_sum < dtype_epsilon<scalar_t>()
                ? dtype_epsilon<scalar_t>()
                : probability_sum;
            scalar_t scale = static_cast<scalar_t>(1) / denominator;
            if (scale > static_cast<scalar_t>(1)) {
              scale = static_cast<scalar_t>(1);
            }
            for (int direction = 0; direction < kDirectionCount; ++direction) {
              probabilities[batch_index * kDirectionCount + direction] *=
                  scale;
            }
          }
        }
      }
    }
    __syncthreads();
    if (abort_sweep != 0) {
      return;
    }

    // Commits are independent across ensembles.  They happen only after the
    // joint strict check above has succeeded.
    for (int64_t batch_index = thread_index; batch_index < batch_size;
         batch_index += blockDim.x) {
      scalar_t cumulative = static_cast<scalar_t>(0);
      int selected_direction = kDirectionCount;
      const scalar_t draw = draws[order_index * batch_size + batch_index];
      for (int direction = 0; direction < kDirectionCount; ++direction) {
        cumulative +=
            probabilities[batch_index * kDirectionCount + direction];
        if (selected_direction == kDirectionCount && draw <= cumulative) {
          selected_direction = direction;
        }
      }

      const int64_t site_offset =
          (batch_index * particle_count + particle_index) * 2;
      const int64_t old_x = sites[site_offset];
      const int64_t old_y = sites[site_offset + 1];
      int64_t chosen_x = old_x;
      int64_t chosen_y = old_y;

      if (selected_direction < kDirectionCount) {
        if (selected_direction == 0) {
          chosen_x = periodic_index(old_x + 1, grid_size);
        } else if (selected_direction == 1) {
          chosen_x = periodic_index(old_x - 1, grid_size);
        } else if (selected_direction == 2) {
          chosen_y = periodic_index(old_y + 1, grid_size);
        } else {
          chosen_y = periodic_index(old_y - 1, grid_size);
        }

        const int64_t direction_offset =
            batch_index * kDirectionCount + selected_direction;
        const int64_t active_offset =
            (batch_index * particle_count + particle_index) *
                kDirectionCount +
            selected_direction;
        const scalar_t active_increment =
            active_work[active_offset] / reservoir_diffusion;
        const scalar_t wca_increment =
            -(mobility * delta_potential[direction_offset]) /
            reservoir_diffusion;
        const scalar_t total_increment = active_increment + wca_increment;
        active_ep[batch_index] += active_increment;
        wca_ep[batch_index] += wca_increment;
        total_ep[batch_index] += total_increment;
        accepted_hops[batch_index] += 1;
        if (record_ep_map) {
          const int64_t map_offset =
              (batch_index * grid_size + old_x) * grid_size + old_y;
          ep_map[map_offset] += total_increment;
        }
      }

      sites[site_offset] = chosen_x;
      sites[site_offset + 1] = chosen_y;
      const int64_t occupancy_offset =
          (batch_index * grid_size + chosen_x) * grid_size + chosen_y;
      occupancy[occupancy_offset] += 1;
    }
    __syncthreads();
  }
}

// Strict production runs use one block per independent ensemble. This exposes
// B-way parallelism to the GPU while retaining the exact particle order and
// exact within-ensemble random-sequential dynamics. If an invalid parameter
// set makes one block fail, other blocks may already have advanced; the Python
// wrapper therefore treats a nonzero status as a failed, unusable sweep.
template <typename scalar_t>
__global__ __launch_bounds__(kThreadCount)
void lattice_abp_sweep_strict_batched_kernel(
    int64_t* __restrict__ sites,
    int64_t* __restrict__ occupancy,
    const int64_t* __restrict__ order,
    const scalar_t* __restrict__ draws,
    const scalar_t* __restrict__ active_work,
    const int64_t* __restrict__ kernel_offsets,
    const scalar_t* __restrict__ kernel_values,
    const int32_t* __restrict__ neighbor_linear,
    int64_t batch_size,
    int64_t particle_count,
    int64_t grid_size,
    int64_t kernel_size,
    scalar_t base,
    scalar_t reservoir_diffusion,
    scalar_t mobility,
    int prefactor_code,
    scalar_t probability_tolerance,
    bool record_ep_map,
    scalar_t* __restrict__ probabilities,
    scalar_t* __restrict__ delta_potential,
    scalar_t* __restrict__ total_ep,
    scalar_t* __restrict__ active_ep,
    scalar_t* __restrict__ wca_ep,
    int64_t* __restrict__ accepted_hops,
    scalar_t* __restrict__ ep_map,
    int32_t* __restrict__ status,
    int64_t* __restrict__ failed_order_index,
    scalar_t* __restrict__ bad_max_sum) {
  const int64_t batch_index = static_cast<int64_t>(blockIdx.x);
  if (batch_index >= batch_size) {
    return;
  }

  __shared__ scalar_t query_energies[kQueryCount];
  __shared__ int64_t query_x[kQueryCount];
  __shared__ int64_t query_y[kQueryCount];
  __shared__ int destination_occupied[kDirectionCount];
  __shared__ int abort_sweep;

  const int thread_index = static_cast<int>(threadIdx.x);
  const int query_index = thread_index / kWarpSize;
  const int lane_index = thread_index % kWarpSize;
  const int64_t probability_base = batch_index * kDirectionCount;

  for (int64_t order_index = 0; order_index < particle_count; ++order_index) {
    const int64_t particle_index = order[order_index];
    if (thread_index == 0) {
      // Ensembles are independent on every valid run. Do not poll the shared
      // error scalar here: that would serialize B blocks on one atomic read
      // per particle. If another block encounters invalid parameters, this
      // block may finish, but the Python wrapper rejects the whole sweep.
      abort_sweep = 0;
      const int64_t site_offset =
          (batch_index * particle_count + particle_index) * 2;
      const int64_t old_x = sites[site_offset];
      const int64_t old_y = sites[site_offset + 1];
      const int64_t old_occupancy_offset =
          (batch_index * grid_size + old_x) * grid_size + old_y;
      occupancy[old_occupancy_offset] -= 1;

      query_x[0] = old_x;
      query_y[0] = old_y;
      query_x[1] = periodic_index(old_x + 1, grid_size);
      query_y[1] = old_y;
      query_x[2] = periodic_index(old_x - 1, grid_size);
      query_y[2] = old_y;
      query_x[3] = old_x;
      query_y[3] = periodic_index(old_y + 1, grid_size);
      query_x[4] = old_x;
      query_y[4] = periodic_index(old_y - 1, grid_size);
      for (int direction = 0; direction < kDirectionCount; ++direction) {
        const int query = direction + 1;
        const int64_t destination_offset =
            (batch_index * grid_size + query_x[query]) * grid_size +
            query_y[query];
        destination_occupied[direction] =
            occupancy[destination_offset] > 0 ? 1 : 0;
      }
    }
    __syncthreads();

    scalar_t energy = static_cast<scalar_t>(0);
    const bool skip_query =
        query_index > 0 && destination_occupied[query_index - 1] != 0;
    if (!skip_query) {
      const int64_t center_x = query_x[query_index];
      const int64_t center_y = query_y[query_index];
      const int64_t center_linear = center_x * grid_size + center_y;
      for (int64_t kernel_index = lane_index; kernel_index < kernel_size;
           kernel_index += kWarpSize) {
        int64_t neighbor_site;
        if (neighbor_linear != nullptr) {
          neighbor_site = static_cast<int64_t>(
              neighbor_linear[center_linear * kernel_size + kernel_index]);
        } else {
          const int64_t neighbor_x = periodic_index(
              center_x + kernel_offsets[kernel_index * 2],
              grid_size);
          const int64_t neighbor_y = periodic_index(
              center_y + kernel_offsets[kernel_index * 2 + 1],
              grid_size);
          neighbor_site = neighbor_x * grid_size + neighbor_y;
        }
        const int64_t occupancy_offset =
            batch_index * grid_size * grid_size + neighbor_site;
        energy += static_cast<scalar_t>(occupancy[occupancy_offset]) *
            kernel_values[kernel_index];
      }
    }
    energy = warp_sum(energy);
    if (lane_index == 0) {
      query_energies[query_index] = energy;
    }
    __syncthreads();

    if (thread_index == 0) {
      bool any_nonfinite = false;
      scalar_t probability_sum = static_cast<scalar_t>(0);
      for (int direction = 0; direction < kDirectionCount; ++direction) {
        const int64_t direction_offset = probability_base + direction;
        const scalar_t energy_change = destination_occupied[direction]
            ? positive_infinity<scalar_t>()
            : query_energies[direction + 1] - query_energies[0];
        delta_potential[direction_offset] = energy_change;
        const int64_t active_offset =
            (batch_index * particle_count + particle_index) *
                kDirectionCount +
            direction;
        const scalar_t probability = transition_probability(
            active_work[active_offset],
            energy_change,
            base,
            reservoir_diffusion,
            mobility,
            prefactor_code);
        probabilities[direction_offset] = probability;
        any_nonfinite = any_nonfinite || !isfinite(probability);
        probability_sum += probability;
      }

      int failure_code = 0;
      if (any_nonfinite) {
        failure_code = 1;
      } else if (
          probability_sum >
          static_cast<scalar_t>(1) + probability_tolerance) {
        failure_code = 2;
      }
      if (failure_code != 0) {
        const int previous = atomicCAS(status, 0, failure_code);
        if (previous == 0) {
          *failed_order_index = order_index;
          if (failure_code == 2) {
            *bad_max_sum = probability_sum;
          }
        }
        // Keep occupancy internally consistent even when the Python caller
        // defers reading status until a save boundary. No site commit has
        // happened yet for this particle.
        const int64_t restore_offset =
            (batch_index * grid_size + query_x[0]) * grid_size + query_y[0];
        occupancy[restore_offset] += 1;
        abort_sweep = 1;
      }
    }
    __syncthreads();
    if (abort_sweep != 0) {
      return;
    }

    if (thread_index == 0) {
      scalar_t cumulative = static_cast<scalar_t>(0);
      int selected_direction = kDirectionCount;
      const scalar_t draw = draws[order_index * batch_size + batch_index];
      for (int direction = 0; direction < kDirectionCount; ++direction) {
        cumulative += probabilities[probability_base + direction];
        if (selected_direction == kDirectionCount && draw <= cumulative) {
          selected_direction = direction;
        }
      }

      const int64_t site_offset =
          (batch_index * particle_count + particle_index) * 2;
      const int64_t old_x = sites[site_offset];
      const int64_t old_y = sites[site_offset + 1];
      int64_t chosen_x = old_x;
      int64_t chosen_y = old_y;
      if (selected_direction < kDirectionCount) {
        if (selected_direction == 0) {
          chosen_x = periodic_index(old_x + 1, grid_size);
        } else if (selected_direction == 1) {
          chosen_x = periodic_index(old_x - 1, grid_size);
        } else if (selected_direction == 2) {
          chosen_y = periodic_index(old_y + 1, grid_size);
        } else {
          chosen_y = periodic_index(old_y - 1, grid_size);
        }

        const int64_t direction_offset =
            probability_base + selected_direction;
        const int64_t active_offset =
            (batch_index * particle_count + particle_index) *
                kDirectionCount +
            selected_direction;
        const scalar_t active_increment =
            active_work[active_offset] / reservoir_diffusion;
        const scalar_t wca_increment =
            -(mobility * delta_potential[direction_offset]) /
            reservoir_diffusion;
        const scalar_t total_increment = active_increment + wca_increment;
        active_ep[batch_index] += active_increment;
        wca_ep[batch_index] += wca_increment;
        total_ep[batch_index] += total_increment;
        accepted_hops[batch_index] += 1;
        if (record_ep_map) {
          const int64_t map_offset =
              (batch_index * grid_size + old_x) * grid_size + old_y;
          ep_map[map_offset] += total_increment;
        }
      }

      sites[site_offset] = chosen_x;
      sites[site_offset + 1] = chosen_y;
      const int64_t chosen_occupancy_offset =
          (batch_index * grid_size + chosen_x) * grid_size + chosen_y;
      occupancy[chosen_occupancy_offset] += 1;
    }
    __syncthreads();
  }
}

}  // namespace

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
    at::Tensor bad_max_sum) {
  const int64_t batch_size = sites.size(0);
  const int64_t particle_count = sites.size(1);
  const int64_t grid_size = occupancy.size(1);
  const int64_t kernel_size = kernel_offsets.size(0);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(sites.get_device());

  AT_DISPATCH_FLOATING_TYPES(
      kernel_values.scalar_type(), "lattice_abp_cuda_sweep", [&] {
        const int32_t* neighbor_ptr = neighbor_linear.numel() > 0
            ? neighbor_linear.data_ptr<int32_t>()
            : nullptr;
        if (strict_probabilities) {
          lattice_abp_sweep_strict_batched_kernel<scalar_t>
              <<<static_cast<unsigned int>(batch_size),
                 kThreadCount,
                 0,
                 stream>>>(
                  sites.data_ptr<int64_t>(),
                  occupancy.data_ptr<int64_t>(),
                  order.data_ptr<int64_t>(),
                  draws.data_ptr<scalar_t>(),
                  active_work.data_ptr<scalar_t>(),
                  kernel_offsets.data_ptr<int64_t>(),
                  kernel_values.data_ptr<scalar_t>(),
                  neighbor_ptr,
                  batch_size,
                  particle_count,
                  grid_size,
                  kernel_size,
                  static_cast<scalar_t>(base),
                  static_cast<scalar_t>(reservoir_diffusion),
                  static_cast<scalar_t>(mobility),
                  static_cast<int>(prefactor_code),
                  static_cast<scalar_t>(probability_tolerance),
                  return_ep_map,
                  probabilities.data_ptr<scalar_t>(),
                  delta_potential.data_ptr<scalar_t>(),
                  total_ep.data_ptr<scalar_t>(),
                  active_ep.data_ptr<scalar_t>(),
                  wca_ep.data_ptr<scalar_t>(),
                  accepted_hops.data_ptr<int64_t>(),
                  return_ep_map ? ep_map.data_ptr<scalar_t>() : nullptr,
                  status.data_ptr<int32_t>(),
                  failed_order_index.data_ptr<int64_t>(),
                  bad_max_sum.data_ptr<scalar_t>());
        } else {
          lattice_abp_sweep_global_kernel<scalar_t>
              <<<1, kThreadCount, 0, stream>>>(
                sites.data_ptr<int64_t>(),
                occupancy.data_ptr<int64_t>(),
                order.data_ptr<int64_t>(),
                draws.data_ptr<scalar_t>(),
                active_work.data_ptr<scalar_t>(),
                kernel_offsets.data_ptr<int64_t>(),
                kernel_values.data_ptr<scalar_t>(),
                neighbor_ptr,
                batch_size,
                particle_count,
                grid_size,
                kernel_size,
                static_cast<scalar_t>(base),
                static_cast<scalar_t>(reservoir_diffusion),
                static_cast<scalar_t>(mobility),
                static_cast<int>(prefactor_code),
                strict_probabilities,
                static_cast<scalar_t>(probability_tolerance),
                return_ep_map,
                probabilities.data_ptr<scalar_t>(),
                delta_potential.data_ptr<scalar_t>(),
                total_ep.data_ptr<scalar_t>(),
                active_ep.data_ptr<scalar_t>(),
                wca_ep.data_ptr<scalar_t>(),
                accepted_hops.data_ptr<int64_t>(),
                return_ep_map ? ep_map.data_ptr<scalar_t>() : nullptr,
                status.data_ptr<int32_t>(),
                failed_order_index.data_ptr<int64_t>(),
                bad_max_sum.data_ptr<scalar_t>());
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
