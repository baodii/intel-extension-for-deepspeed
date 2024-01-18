// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cassert>
#include "activation_type.h"
#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"

// Default activation function will error out
template <ActivationType ActType>
DS_D_INLINE float act_fn(float val);

template <>
DS_D_INLINE float act_fn<ActivationType::IDENTITY>(float val)
{
    return val;
}

template <>
DS_D_INLINE float act_fn<ActivationType::RELU>(float val)
{
    return val > 0.0f ? val : 0.0f;
}

template <>
DS_D_INLINE float act_fn<ActivationType::GELU>(float val)
{
    constexpr float sqrt_param = 0.79788456080286535587989211986876f;
    constexpr float mul_param = 0.044715f;
    return val * 0.5f * (1.0f + sycl::tanh(sqrt_param * (val + mul_param * val * val * val)));
}

template <>
DS_D_INLINE float act_fn<ActivationType::SILU>(float val)
{
    return val / (1.0f + sycl::native::exp(-val));
}

namespace bias_act {

constexpr int access_size = 16;
constexpr int threads = 512;
constexpr int unroll = 4;

}  // namespace bias_act

template <typename T, ActivationType ActType>
void bias_activation_kernel(T* activation,
                                       const T* bias,
                                       const int32_t rows,
                                       const int32_t cols)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int vector_T = bias_act::access_size / sizeof(T);

    const int32_t thread_offset = item_ct1.get_local_id(2) * vector_T;
    const int32_t block_offset =
        item_ct1.get_group(2) * vector_T * bias_act::unroll * bias_act::threads;
    const int32_t base_offset = block_offset + thread_offset;

    const int32_t thread_stride = bias_act::threads * vector_T;

#pragma unroll
    for (int i = 0; i < bias_act::unroll; i++) {
        const int32_t iter_offset = base_offset + i * thread_stride;

        const int32_t row = iter_offset / cols;

        T buffer[vector_T];
        T bias_buffer[vector_T];

        if (row < rows) {
            const int32_t col = iter_offset % cols;

            mem_access::load_global<bias_act::access_size>(buffer, activation + iter_offset);
            mem_access::load_global<bias_act::access_size>(
                bias_buffer, bias + col, bias != nullptr);

#pragma unroll
            for (int j = 0; j < vector_T; j++) {
                float val =
                    conversion::to<float>(buffer[j]) + conversion::to<float>(bias_buffer[j]);
                buffer[j] = conversion::to<T>(act_fn<ActType>(val));
            }

            mem_access::store_global<bias_act::access_size>(activation + iter_offset, buffer);
        }
    }
}

#define ACT_TYPE_SWITCH(ACT_TYPE, ...)                                \
    if (ACT_TYPE == ActivationType::IDENTITY) {                       \
        constexpr ActivationType act_fn_t = ActivationType::IDENTITY; \
        return __VA_ARGS__();                                         \
    } else if (ACT_TYPE == ActivationType::RELU) {                    \
        constexpr ActivationType act_fn_t = ActivationType::RELU;     \
        return __VA_ARGS__();                                         \
    } else if (ACT_TYPE == ActivationType::GELU) {                    \
        constexpr ActivationType act_fn_t = ActivationType::GELU;     \
        return __VA_ARGS__();                                         \
    } else if (ACT_TYPE == ActivationType::SILU) {                    \
        constexpr ActivationType act_fn_t = ActivationType::SILU;     \
        return __VA_ARGS__();                                         \
    } else {                                                          \
        assert(false);                                                \
    }

template <typename T>
void launch_bias_activation(T* activation,
                            const T* bias,
                            const int32_t n_rows,
                            const int32_t n_cols,
                            const ActivationType activation_type,
                            dpct::queue_ptr stream)
{
    constexpr int32_t elems_per_block =
        bias_act::threads * bias_act::unroll * bias_act::access_size / sizeof(T);
    const int32_t total_elems = n_rows * n_cols;

    const int32_t blocks = (total_elems + elems_per_block - 1) / elems_per_block;

    const sycl::range<3> grid(1, 1, blocks);
    const sycl::range<3> block(1, 1, bias_act::threads);

    /*
    DPCT1038:10: When the kernel function name is used as a macro argument, the migration result may
    be incorrect. You need to verify the definition of the macro.
    */
    ACT_TYPE_SWITCH(activation_type, [&] {
        /*
        DPCT1049:11: The work-group size passed to the SYCL kernel may exceed the limit. To get the
        device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
        */
  dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});

  stream->parallel_for(sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
   bias_activation_kernel<T, act_fn_t>(activation, bias, n_rows, n_cols);
  });
    });
}

#define INSTANTIATE_FOR_T(T)                 \
    template void launch_bias_activation<T>( \
        T*, const T*, const int32_t, const int32_t, const ActivationType, dpct::queue_ptr);

INSTANTIATE_FOR_T(sycl::half);

#ifdef BF16_AVAILABLE
INSTANTIATE_FOR_T(sycl::ext::oneapi::bfloat16);
#endif
