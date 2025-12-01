#pragma once

#include <cstddef>
#include <cstdint>

#if defined(USE_ROCM)
#include <hip/hip_runtime_api.h>
#endif

namespace c10d::symmetric_memory {

// Covers NVL72
constexpr int max_cuda_p2p_domain_size = 72;
// Maximum number of channels
constexpr int symm_max_nblocks = 32;

// Maximally, a rank will need to sync with all other ranks, over all
// channels. Each signal is 32 bits, which is the minimum unit for atomic cas.
// Default signal pad size, can be overridden via set_signal_pad_size().
constexpr size_t default_signal_pad_size =
    symm_max_nblocks * max_cuda_p2p_domain_size * sizeof(uint32_t);

// Get the current signal pad size. Returns the user-configured size if set,
// otherwise returns the default size.
size_t get_signal_pad_size();

// Set the signal pad size for future symmetric memory allocations.
// This must be called before any symmetric memory allocations are made.
// The size should be proportional to the number of blocks the user launches
// and the world size.
void set_signal_pad_size(size_t size);

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
using HandleType = CUmemGenericAllocationHandle;
#elif defined(USE_ROCM)
using HandleType = hipMemGenericAllocationHandle_t;
#else
using HandleType = void*;
#endif

} // namespace c10d::symmetric_memory
