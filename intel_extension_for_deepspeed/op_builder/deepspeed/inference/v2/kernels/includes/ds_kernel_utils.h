// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Centralized header file for preprocessor macros and constants
used throughout the codebase.
*/

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <ipex.h>

#ifdef BF16_AVAILABLE
#endif

#define DS_HD_INLINE __forceinline__
#define DS_D_INLINE __dpct_inline__

#ifdef __HIP_PLATFORM_AMD__

// constexpr variant of warpSize for templating
constexpr int hw_warp_size = 64;
#define HALF_PRECISION_AVAILABLE = 1
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>

#else  // !__HIP_PLATFORM_AMD__

// constexpr variant of warpSize for templating
constexpr int hw_warp_size = 32;

#if DPCT_COMPATIBILITY_TEMP >= 530
#define HALF_PRECISION_AVAILABLE = 1
/* #define PTX_AVAILABLE */
#endif  // __CUDA_ARCH__ >= 530

#if 0
#define ASYNC_COPY_AVAILABLE
#endif  // __CUDA_ARCH__ >= 800

#endif  //__HIP_PLATFORM_AMD__

namespace at {
  namespace cuda {
    dpct::queue_ptr getCurrentCUDAStream(); 

    dpct::queue_ptr getStreamFromPool(bool); 
  }
}

inline int next_pow2(const int val)
{
    int rounded_val = val - 1;
    rounded_val |= rounded_val >> 1;
    rounded_val |= rounded_val >> 2;
    rounded_val |= rounded_val >> 4;
    rounded_val |= rounded_val >> 8;
    return rounded_val + 1;
}
