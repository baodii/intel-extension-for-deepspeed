// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ds_kernel_utils.h"
#include "reduction_utils.h"
#include "top_k_gating.dp.hpp"
#include "top_k_utils.h"

using ROp = reduce::ROpType;

namespace scatter {

constexpr int access_granularity = 16;
constexpr int threads = 256;
constexpr int warps = threads / hw_warp_size;
constexpr int max_experts = 1024;

}  // namespace scatter

template <typename T, int copyUnroll, int N_TOP_K>
/*
DPCT1110:13: The total declared local variable size in device function moe_scatter_kernel exceeds
128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total
register size available and adjust the code, or use smaller sub-group size to avoid high register
pressure.
*/
void moe_scatter_kernel(T* moe_input,
                        int64_t* expert_count_cumsums,
                        int32_t* mapped_slots,
                        const T* activations,
                        const int32_t* assignments,
                        const int32_t* expert_counts,
                        const int32_t* offsets,
                        const int32_t n_channels,
                        const int32_t n_experts)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int32_t vector_size = scatter::access_granularity / sizeof(T);
    constexpr int32_t load_stride = vector_size * scatter::threads;

    const int32_t token_idx = item_ct1.get_group(2);
    const int32_t tidx = item_ct1.get_local_id(2);
    const int32_t warp_rank = tidx / hw_warp_size;

    // Bank aligned and sufficient
    auto& red_buffer = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[32]>(
        sycl::ext::oneapi::experimental::this_group<3>());
    auto& expert_offsets =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[scatter::max_experts]>(
            sycl::ext::oneapi::experimental::this_group<3>());

    // CG helpers
    sycl::group<3> tb = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::experimental::this_sub_group();

    // Fetch the assigned experts for this token.
    int assigned_experts[N_TOP_K];
    for (int i = 0; i < N_TOP_K; i++) {
        assigned_experts[i] = assignments[token_idx * N_TOP_K + i];
    }

    bool all_unassigned = true;
    for (int i = 0; i < N_TOP_K; i++) {
        if (assigned_experts[i] != gating::unassigned) {
            all_unassigned = false;
        } else {
            mapped_slots[token_idx * N_TOP_K + i] = gating::unassigned;
        }
    }
    if (all_unassigned && token_idx != 0) return;

    // Do a prefix scan on the expert counts to get the base offsets. Here we use the
    // single up-sweep variant.
    int32_t expert_vals;
    if (tidx < n_experts) {
        expert_vals = expert_counts[tidx];
    } else {
        expert_vals = 0;
    }

#pragma unroll
    for (int i = 1; i < hw_warp_size; i *= 2) {
        int32_t maybe_add = sycl::shift_group_right(
            sycl::ext::oneapi::experimental::this_sub_group(), expert_vals, i);
        expert_vals = (sycl::ext::oneapi::experimental::this_sub_group().get_local_linear_id() < i)
                          ? expert_vals
                          : expert_vals + maybe_add;
    }

    if (sycl::ext::oneapi::experimental::this_sub_group().get_local_linear_id() ==
        hw_warp_size - 1) {
        mem_access::store_shared<4>(red_buffer + warp_rank, &expert_vals);
    }

    /*
    DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there
    is no access to global memory.
    */
    item_ct1.barrier();

    int32_t phase_2_val = 0;
    if (sycl::ext::oneapi::experimental::this_sub_group().get_local_linear_id() < scatter::warps) {
        mem_access::load_shared<4>(
            &phase_2_val,
            red_buffer + sycl::ext::oneapi::experimental::this_sub_group().get_local_linear_id());
    }

#pragma unroll
    for (int i = 1; i < hw_warp_size; i *= 2) {
        int32_t maybe_add = sycl::shift_group_right(
            sycl::ext::oneapi::experimental::this_sub_group(), phase_2_val, i);
        phase_2_val = (sycl::ext::oneapi::experimental::this_sub_group().get_local_linear_id() < i)
                          ? phase_2_val
                          : phase_2_val + maybe_add;
    }

    int warp_offset = 0;
    if (warp_rank > 0) {
        warp_offset = sycl::select_from_group(
            sycl::ext::oneapi::experimental::this_sub_group(), phase_2_val, warp_rank - 1);
    }
    const int32_t expert_cumsum = warp_offset + expert_vals;

    // Token 0 will write the
    if (token_idx == 0 && tidx < n_experts) {
        int64_t expert_cumsum_64 = (int64_t)expert_cumsum;
        expert_count_cumsums[tidx] = expert_cumsum_64;
    }

    // Since token 0 has now written the expert cumsum to global memory,
    // if it has no valid experts, we can early return.
    if (token_idx == 0 && all_unassigned) return;

    if (tidx < n_experts) { expert_offsets[tidx] = expert_cumsum; }

    // Ensure all the expert offsets are written in shared memory.
    /*
    DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there
    is no access to global memory.
    */
    item_ct1.barrier();

    // Data copy to appropriate location
    const int32_t thread_offset = tidx * vector_size;

    const int32_t base_load_offset = token_idx * n_channels + thread_offset;
    const T* load_base_ptr = activations + base_load_offset;

    int32_t store_rows[N_TOP_K];
    T* store_base_ptrs[N_TOP_K];
#pragma unroll
    for (int i = 0; i < N_TOP_K; i++) {
        const int32_t cur_expert_offset =
            (assigned_experts[i] > 0) ? expert_offsets[assigned_experts[i] - 1] : 0;
        store_rows[i] = cur_expert_offset + offsets[token_idx * N_TOP_K + i];
        const int32_t base_store_offset = store_rows[i] * n_channels + thread_offset;
        store_base_ptrs[i] = moe_input + base_store_offset;
    }

#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {
        T tmp_buf[vector_size];

        if (i * load_stride + thread_offset < n_channels) {
            mem_access::load_global<scatter::access_granularity>(tmp_buf,
                                                                 load_base_ptr + i * load_stride);
#pragma unroll
            for (int j = 0; j < N_TOP_K; j++) {
                mem_access::store_global<scatter::access_granularity>(
                    store_base_ptrs[j] + i * load_stride, tmp_buf);
            }
        }
    }

    if (item_ct1.get_local_id(2) == 0) {
        for (int i = 0; i < N_TOP_K; i++) { mapped_slots[token_idx * N_TOP_K + i] = store_rows[i]; }
    }
}

/*
DPCT1049:17: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define LAUNCH_FOR_UNROLL(COUNT)                                                                 \
    case COUNT: {                                                                                \
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});                \
                                                                                                 \
        stream->submit([&](sycl::handler& cgh) {                                                 \
            T* moe_input_ct0 = moe_input;                                                        \
            int64_t* expert_count_cumsums_ct1 = expert_count_cumsums;                            \
            int32_t* mapped_slots_ct2 = mapped_slots;                                            \
            const T* activations_ct3 = activations;                                              \
            const int32_t* assignments_ct4 = assignments;                                        \
            const int32_t* expert_counts_ct5 = expert_counts;                                    \
            const int32_t* offsets_ct6 = offsets;                                                \
            auto n_channels_ct7 = n_channels;                                                    \
            auto n_experts_ct8 = n_experts;                                                      \
                                                                                                 \
            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                             \
                             [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] { \
                                 moe_scatter_kernel<T, COUNT, CONST_TOP_K>(                      \
                                     moe_input_ct0,                                              \
                                     expert_count_cumsums_ct1,                                   \
                                     mapped_slots_ct2,                                           \
                                     activations_ct3,                                            \
                                     assignments_ct4,                                            \
                                     expert_counts_ct5,                                          \
                                     offsets_ct6,                                                \
                                     n_channels_ct7,                                             \
                                     n_experts_ct8);                                             \
                             });                                                                 \
        });                                                                                      \
    } break;

template <typename T>
void launch_moe_scatter(T* moe_input,
                        int64_t* expert_count_cumsums,
                        int32_t* mapped_slots,
                        const T* activations,
                        const int32_t* expert_counts,
                        const int32_t* assignments,
                        const int32_t* offsets,
                        const int32_t n_channels,
                        const int32_t n_tokens,
                        const int32_t n_experts,
                        const int32_t n_top_k,
                        dpct::queue_ptr stream)
{
    constexpr int vals_per_unroll = scatter::threads * scatter::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const sycl::range<3> block(1, 1, scatter::threads);
    const sycl::range<3> grid(1, 1, n_tokens);

    /*
    DPCT1038:16: When the kernel function name is used as a macro argument, the migration result may
    be incorrect. You need to verify the definition of the macro.
    */
    TOP_K_SWITCH(n_top_k, [&] {
        switch (copy_unroll) {
            LAUNCH_FOR_UNROLL(1);
            LAUNCH_FOR_UNROLL(2);
            LAUNCH_FOR_UNROLL(3);
            LAUNCH_FOR_UNROLL(4);
            LAUNCH_FOR_UNROLL(5);
            LAUNCH_FOR_UNROLL(6);
        }
    });
}

#define INSTANTIATE_SCATTER_FOR_TYPE(TYPE)                 \
    template void launch_moe_scatter<TYPE>(TYPE*,          \
                                           int64_t*,       \
                                           int32_t*,       \
                                           const TYPE*,    \
                                           const int32_t*, \
                                           const int32_t*, \
                                           const int32_t*, \
                                           const int32_t,  \
                                           const int32_t,  \
                                           const int32_t,  \
                                           const int32_t,  \
                                           dpct::queue_ptr);

INSTANTIATE_SCATTER_FOR_TYPE(sycl::half);

#ifdef BF16_AVAILABLE
INSTANTIATE_SCATTER_FOR_TYPE(sycl::ext::oneapi::bfloat16);
#endif
