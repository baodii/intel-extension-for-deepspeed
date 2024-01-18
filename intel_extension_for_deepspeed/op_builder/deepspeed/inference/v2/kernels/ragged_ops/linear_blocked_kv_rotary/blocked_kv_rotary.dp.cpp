// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "blocked_kv_rotary.dp.hpp"
#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"

namespace kv_rot {

constexpr int granularity = 16;
constexpr int threads = 256;

}  // namespace kv_rot

/*
Supports head size 32, 64, 128, 256
*/

template <typename T, int qRatio, int headSize, bool doRotary, int paddedHeadSize>
/*
DPCT1110:10: The total declared local variable size in device function kv_rotary_pos_kernel exceeds
128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total
register size available and adjust the code, or use smaller sub-group size to avoid high register
pressure.
*/
void kv_rotary_pos_kernel(T* kv_cache,
                          T* q,
                          T* k,
                          T* v,
                          const T* inv_freq,
                          const int32_t rotary_dim,
                          const float theta_base,
                          const BatchWrapperCPP batch_desc,
                          const int qkv_stride,
                          const int kv_cache_stride,
                          const int v_offset,
                          const int inv_freq_stride)
{
    // Derived constexpr
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int vector_T = kv_rot::granularity / sizeof(T);
    constexpr int real_threads_per_head = headSize / vector_T;
    constexpr int threads_per_head = paddedHeadSize / vector_T;

    constexpr int tokens_per_block = kv_rot::threads / threads_per_head;

    // CG helpers
    sycl::group<3> tb = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::experimental::this_sub_group();
    /*
    DPCT1007:11: Migration of tiled_partition is not supported.
    */
    auto head_group = sycl::ext::oneapi::experimental::this_sub_group();

    // Parallelize on the head dimension for X blocks
    const int head_idx = item_ct1.get_group(2);

    const int block_seq_idx = item_ct1.get_local_id(2) / threads_per_head;
    /*
    DPCT1007:12: Migration of thread_rank is not supported.
    */
    const int base_neuron_idx = head_group.get_local_linear_id() * vector_T;
    const int half_rotary_size = rotary_dim / 2;
    const int half_dim_lanes = half_rotary_size / vector_T;
    const int half_idx = base_neuron_idx % half_rotary_size;

    // Multiple tokens processed by the same threadblock
    const int token_idx = item_ct1.get_group(1) * tokens_per_block + block_seq_idx;
    const bool valid_token = token_idx < batch_desc.batch_metadata->n_tokens;

    /*
    DPCT1007:13: Migration of thread_rank is not supported.
    */
    const bool valid_thread = valid_token && (head_group.get_local_linear_id() < real_threads_per_head);
    const bool load_inv_freq = (inv_freq != nullptr) && valid_thread;

    // If we have GQA, then only one of the Q heads needs to do rotary + copy
    // for each of the heads in the group.
    bool need_kv = head_idx % qRatio == 0;
    // Make sure the following code is warp uniform
    need_kv =
        sycl::select_from_group(sycl::ext::oneapi::experimental::this_sub_group(), need_kv, 0);

    const int kv_head_idx = head_idx / qRatio;

    // Ensure we don't access invalid portions of the seq_metadata
    const int32_t seq_id = (valid_thread) ? batch_desc.tokens_to_seq[token_idx] : 0;
    const InflightSeqDescriptor seq_desc = batch_desc.seq_metadata[seq_id];
    // This will give an invalid index if valid_thread is false, but should never affect memory.
    const int32_t global_token_idx = seq_desc.seen_tokens + (token_idx - seq_desc.start_idx);

    T* q_row = q + token_idx * qkv_stride + head_idx * headSize;
    T q_reg[vector_T];

    if (need_kv) {
        // The following logic assumes a linearly blocked KV cache. This means that no sparsity has
        // been introduced into cache history.
        const KVCacheDescriptor kv_desc = batch_desc.kv_desc;
        const int32_t seq_kv_block_idx = global_token_idx / kv_desc.block_size;
        const int32_t mapped_kv_block_idx =
            (valid_thread) ? kv_desc.block_lists[seq_id][seq_kv_block_idx] : 0;

        const int32_t kv_block_offset = global_token_idx % kv_desc.block_size;
        const int32_t kv_offset =
            (mapped_kv_block_idx * kv_desc.block_size + kv_block_offset) * kv_cache_stride +
            kv_head_idx * headSize;

        // Load indices from QKV output
        T* k_row = k + token_idx * qkv_stride + kv_head_idx * headSize;
        T* v_row = v + token_idx * qkv_stride + kv_head_idx * headSize;

        T k_reg[vector_T], v_reg[vector_T], inv_freq_reg[vector_T];

        mem_access::load_global<kv_rot::granularity>(q_reg, q_row + base_neuron_idx, valid_thread);
        mem_access::load_global<kv_rot::granularity>(k_reg, k_row + base_neuron_idx, valid_thread);
        mem_access::load_global<kv_rot::granularity>(v_reg, v_row + base_neuron_idx, valid_thread);
        mem_access::load_global<kv_rot::granularity>(
            inv_freq_reg, inv_freq + half_idx, load_inv_freq);
        if constexpr (doRotary) {
#pragma unroll
            for (int i = 0; i < vector_T; i++) {
                const int head_neuron_idx = base_neuron_idx + i;

                float inv_freq_flt;
                if (inv_freq != nullptr) {
                    inv_freq_flt = conversion::to<float>(inv_freq_reg[i]) * (float)global_token_idx;
                } else {
                    inv_freq_flt =
                        (float)((head_neuron_idx % half_rotary_size) * 2) / (float)rotary_dim;
                    // Conversion to T and back means that both branches of this if statement
                    // will produce the same results if using the same algo for producing the
                    // freqs.
                    T trunc_freq = conversion::to<T>(1.0 / dpct::pow(theta_base, inv_freq_flt));
                    inv_freq_flt = conversion::to<float>(trunc_freq) * (float)global_token_idx;
                }

                float rotary_sign = (head_neuron_idx >= half_rotary_size) ? -1.0f : 1.0f;
                float q_f = conversion::to<float>(q_reg[i]);
                float k_f = conversion::to<float>(k_reg[i]);
                float q_rot = q_f * rotary_sign;
                float k_rot = k_f * rotary_sign;

                const int target_lane = (head_neuron_idx < half_rotary_size)
                                            /*
                                            DPCT1007:14: Migration of thread_rank is not supported.
                                            */
                                            ? head_group.get_local_linear_id() + half_dim_lanes
                                            /*
                                            DPCT1007:15: Migration of thread_rank is not supported.
                                            */
                                            : head_group.get_local_linear_id() - half_dim_lanes;

                const float q_rot_temp = dpct::select_from_sub_group(
                    sycl::ext::oneapi::experimental::this_sub_group(), q_rot, target_lane, 8);
                const float k_rot_temp = dpct::select_from_sub_group(
                    sycl::ext::oneapi::experimental::this_sub_group(), k_rot, target_lane, 8);

                if (base_neuron_idx < rotary_dim) {
                    q_reg[i] = conversion::to<T>(q_f * sycl::cos(inv_freq_flt) +
                                                 q_rot_temp * sycl::sin(inv_freq_flt));
                    k_reg[i] = conversion::to<T>(k_f * sycl::cos(inv_freq_flt) +
                                                 k_rot_temp * sycl::sin(inv_freq_flt));
                }
            }
        }

        if (valid_thread) {
            mem_access::store_global<kv_rot::granularity>(kv_cache + kv_offset + base_neuron_idx,
                                                          k_reg);
            mem_access::store_global<kv_rot::granularity>(
                kv_cache + kv_offset + base_neuron_idx + v_offset, v_reg);
        }
    } else {
        T inv_freq_reg[vector_T];

        mem_access::load_global<kv_rot::granularity>(q_reg, q_row + base_neuron_idx, valid_thread);
        mem_access::load_global<kv_rot::granularity>(
            inv_freq_reg, inv_freq + half_idx, load_inv_freq);

        if constexpr (doRotary) {
#pragma unroll
            for (int i = 0; i < vector_T; i++) {
                const int head_neuron_idx = base_neuron_idx + i;

                float inv_freq_flt;
                if (inv_freq != nullptr) {
                    inv_freq_flt = conversion::to<float>(inv_freq_reg[i]) * (float)global_token_idx;
                } else {
                    inv_freq_flt =
                        (float)((head_neuron_idx % half_rotary_size) * 2) / (float)rotary_dim;
                    inv_freq_flt =
                        1.0 / dpct::pow(theta_base, inv_freq_flt) * (float)global_token_idx;
                }

                float rotary_sign = (head_neuron_idx >= half_rotary_size) ? -1.0f : 1.0f;
                float q_f = conversion::to<float>(q_reg[i]);
                float q_rot = q_f * rotary_sign;

                const int target_lane = (head_neuron_idx < half_rotary_size)
                                            /*
                                            DPCT1007:16: Migration of thread_rank is not supported.
                                            */
                                            ? head_group.get_local_linear_id() + half_dim_lanes
                                            /*
                                            DPCT1007:17: Migration of thread_rank is not supported.
                                            */
                                            : head_group.get_local_linear_id() - half_dim_lanes;

                const float q_rot_temp = dpct::select_from_sub_group(
                    sycl::ext::oneapi::experimental::this_sub_group(), q_rot, target_lane, 8);
                if (base_neuron_idx < rotary_dim)
                    q_reg[i] = conversion::to<T>(q_f * sycl::cos(inv_freq_flt) +
                                                 q_rot_temp * sycl::sin(inv_freq_flt));
            }
        }
    }

    if (valid_thread && doRotary) {
        mem_access::store_global<kv_rot::granularity>(q_row + base_neuron_idx, q_reg);
    }
}

/*
DPCT1049:18: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define DISPATCH_KV_ROTARY_IMPL(Q_RATIO, HEAD_SIZE, PADDED_HEAD_SIZE)                           \
 if (q_ratio == Q_RATIO && head_size == HEAD_SIZE)                                              \
 {                                                                                              \
  dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16}); \
                                                                                                \
  stream->submit([&](sycl::handler& cgh) {                                                      \
   T* kv_cache_ct0 = kv_cache;                                                                  \
   T* q_ct1 = q;                                                                                \
   T* k_ct2 = k;                                                                                \
   T* v_ct3 = v;                                                                                \
   T* inv_freq_ct4 = inv_freq;                                                                  \
   auto rotary_dim_ct5 = rotary_dim;                                                            \
   auto theta_base_ct6 = theta_base;                                                            \
   auto batch_desc_ct0 = batch_desc;                                                            \
   auto qkv_stride_ct8 = qkv_stride;                                                            \
   auto kv_cache_stride_ct9 = kv_cache_stride;                                                  \
   auto v_offset_ct10 = v_offset;                                                               \
   auto inv_freq_stride_ct11 = inv_freq_stride;                                                 \
                                                                                                \
   cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                                     \
                    [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {         \
                     kv_rotary_pos_kernel<T, Q_RATIO, HEAD_SIZE, true, PADDED_HEAD_SIZE>(       \
                         kv_cache_ct0,                                                          \
                         q_ct1,                                                                 \
                         k_ct2,                                                                 \
                         v_ct3,                                                                 \
                         inv_freq_ct4,                                                          \
                         rotary_dim_ct5,                                                        \
                         theta_base_ct6,                                                        \
                         batch_desc_ct0,                                                        \
                         qkv_stride_ct8,                                                        \
                         kv_cache_stride_ct9,                                                   \
                         v_offset_ct10,                                                         \
                         inv_freq_stride_ct11);                                                 \
                    });                                                                         \
  });                                                                                           \
 }

#define LAUNCH_KV_ROTARY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, HEAD_SIZE) \
    if (padded_head_size == 64) {                                  \
        DISPATCH_KV_ROTARY_IMPL(Q_RATIO, HEAD_SIZE, 64);           \
    } else if (padded_head_size == 128) {                          \
        DISPATCH_KV_ROTARY_IMPL(Q_RATIO, HEAD_SIZE, 128);          \
    } else {                                                       \
        assert(false);                                             \
    }

#define LAUNCH_KV_ROTARY_FOR_Q_RATIO(Q_RATIO)                 \
    if (head_size == 64) {                                    \
        LAUNCH_KV_ROTARY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, 64);  \
    } else if (head_size == 80) {                             \
        LAUNCH_KV_ROTARY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, 80);  \
    } else if (head_size == 128) {                            \
        LAUNCH_KV_ROTARY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, 128); \
    } else {                                                  \
        assert(false);                                        \
    }

template <typename T>
void launch_kv_rotary_kernel(T* kv_cache,
                             T* q,
                             T* k,
                             T* v,
                             T* inv_freq,
                             const int32_t rotary_dim,
                             const float theta_base,
                             const BatchWrapperCPP batch_desc,
                             const int qkv_stride,
                             const int kv_cache_stride,
                             const int v_offset,
                             const int inv_freq_stride,
                             const int q_ratio,
                             const int head_size,
                             const int n_tokens,
                             const int n_q_heads,
                             dpct::queue_ptr stream)
{
    constexpr int vector_T = kv_rot::granularity / sizeof(T);

    const int padded_head_size = next_pow2(head_size);
    const int threads_per_head = padded_head_size / vector_T;

    const int tokens_per_block = kv_rot::threads / threads_per_head;

    const sycl::range<3> block(1, 1, kv_rot::threads);
    const int token_blocks = (n_tokens + tokens_per_block - 1) / tokens_per_block;
    const sycl::range<3> grid(1, token_blocks, n_q_heads);

    LAUNCH_KV_ROTARY_FOR_Q_RATIO(1)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(2)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(4)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(5)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(8)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(16)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(29)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(35)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(36)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(71)
}

#define INSTANTIATE_KV_ROTARY_KERNEL(TYPE)                                        \
    template void launch_kv_rotary_kernel<TYPE>(TYPE * kv_cache,                  \
                                                TYPE * q,                         \
                                                TYPE * k,                         \
                                                TYPE * v,                         \
                                                TYPE * inv_freq,                  \
                                                const int32_t rotary_dim,         \
                                                const float theta_base,           \
                                                const BatchWrapperCPP batch_desc, \
                                                const int qkv_stride,             \
                                                const int kv_cache_stride,        \
                                                const int v_offset,               \
                                                const int inv_freq_stride,        \
                                                const int q_ratio,                \
                                                const int head_size,              \
                                                const int n_tokens,               \
                                                const int n_q_heads,              \
                                                dpct::queue_ptr stream);

INSTANTIATE_KV_ROTARY_KERNEL(sycl::half)

#ifdef BF16_AVAILABLE
INSTANTIATE_KV_ROTARY_KERNEL(sycl::ext::oneapi::bfloat16)
#endif

/*
DPCT1049:19: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define DISPATCH_KV_COPY_IMPL(Q_RATIO, HEAD_SIZE, PADDED_HEAD_SIZE)                             \
 if (q_ratio == Q_RATIO && head_size == HEAD_SIZE)                                              \
 {                                                                                              \
  dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16}); \
                                                                                                \
  stream->submit([&](sycl::handler& cgh) {                                                      \
   T* kv_cache_ct0 = kv_cache;                                                                  \
   T* q_ct1 = q;                                                                                \
   T* k_ct2 = k;                                                                                \
   T* v_ct3 = v;                                                                                \
   auto nullptr_ct4 = nullptr;                                                                  \
   auto ct5 = -1;                                                                               \
   auto ct6 = 0.f;                                                                              \
   auto batch_desc_ct0 = batch_desc;                                                            \
   auto qkv_stride_ct8 = qkv_stride;                                                            \
   auto kv_cache_stride_ct9 = kv_cache_stride;                                                  \
   auto v_offset_ct10 = v_offset;                                                               \
   auto ct11 = 0;                                                                               \
                                                                                                \
   cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                                     \
                    [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {         \
                     kv_rotary_pos_kernel<T, Q_RATIO, HEAD_SIZE, false, PADDED_HEAD_SIZE>(      \
                         kv_cache_ct0,                                                          \
                         q_ct1,                                                                 \
                         k_ct2,                                                                 \
                         v_ct3,                                                                 \
                         nullptr_ct4,                                                           \
                         ct5,                                                                   \
                         ct6,                                                                   \
                         batch_desc_ct0,                                                        \
                         qkv_stride_ct8,                                                        \
                         kv_cache_stride_ct9,                                                   \
                         v_offset_ct10,                                                         \
                         ct11);                                                                 \
                    });                                                                         \
  });                                                                                           \
 }

#define LAUNCH_KV_COPY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, HEAD_SIZE) \
    if (padded_head_size == 64) {                                \
        DISPATCH_KV_COPY_IMPL(Q_RATIO, HEAD_SIZE, 64);           \
    } else if (padded_head_size == 128) {                        \
        DISPATCH_KV_COPY_IMPL(Q_RATIO, HEAD_SIZE, 128);          \
    } else {                                                     \
        assert(false);                                           \
    }

#define LAUNCH_KV_COPY_FOR_Q_RATIO(Q_RATIO)                 \
    if (head_size == 64) {                                  \
        LAUNCH_KV_COPY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, 64);  \
    } else if (head_size == 80) {                           \
        LAUNCH_KV_COPY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, 80);  \
    } else if (head_size == 128) {                          \
        LAUNCH_KV_COPY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, 128); \
    } else {                                                \
        assert(false);                                      \
    }

template <typename T>
void launch_kv_copy_kernel(T* kv_cache,
                           T* q,
                           T* k,
                           T* v,
                           const BatchWrapperCPP batch_desc,
                           const int qkv_stride,
                           const int kv_cache_stride,
                           const int v_offset,
                           const int q_ratio,
                           const int head_size,
                           const int n_tokens,
                           const int n_q_heads,
                           dpct::queue_ptr stream)
{
    constexpr int vector_T = kv_rot::granularity / sizeof(T);
    const int padded_head_size = next_pow2(head_size);
    const int threads_per_head = padded_head_size / vector_T;
    const int tokens_per_block = kv_rot::threads / threads_per_head;

    const sycl::range<3> block(1, 1, kv_rot::threads);
    const int token_blocks = (n_tokens + tokens_per_block - 1) / tokens_per_block;
    const sycl::range<3> grid(1, token_blocks, n_q_heads);

    LAUNCH_KV_COPY_FOR_Q_RATIO(1)
    LAUNCH_KV_COPY_FOR_Q_RATIO(2)
    LAUNCH_KV_COPY_FOR_Q_RATIO(4)
    LAUNCH_KV_COPY_FOR_Q_RATIO(5)
    LAUNCH_KV_COPY_FOR_Q_RATIO(8)
}

#define INSTANTIATE_KV_COPY_KERNEL(TYPE)                                        \
    template void launch_kv_copy_kernel<TYPE>(TYPE * kv_cache,                  \
                                              TYPE * q,                         \
                                              TYPE * k,                         \
                                              TYPE * v,                         \
                                              const BatchWrapperCPP batch_desc, \
                                              const int qkv_stride,             \
                                              const int kv_cache_stride,        \
                                              const int v_offset,               \
                                              const int q_ratio,                \
                                              const int head_size,              \
                                              const int n_tokens,               \
                                              const int n_q_heads,              \
                                              dpct::queue_ptr stream);

INSTANTIATE_KV_COPY_KERNEL(sycl::half)

#ifdef BF16_AVAILABLE
INSTANTIATE_KV_COPY_KERNEL(sycl::ext::oneapi::bfloat16)
#endif
