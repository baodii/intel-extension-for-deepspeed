// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "conversion_utils.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"
#include "top_k_gating.dp.hpp"
#include "top_k_utils.h"
#include <cmath>

using ROp = reduce::ROpType;

template <typename T, int TOP_K>
void top_k_gating_kernel(int32_t* expert_counts,
                                    float* scores,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    const T* logits,
                                    const RaggedBatchDescriptor* batch_metadata,
                                    const int32_t n_experts)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    const int32_t token_idx = item_ct1.get_group(2);
    const int32_t expert_idx = item_ct1.get_local_id(2);
    const int32_t max_warps = 1024 / hw_warp_size;

    // CG helpers
    sycl::group<3> tb = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::experimental::this_sub_group();

    // Padding tokens do not require
    if (token_idx >= batch_metadata->n_tokens) {
        if (item_ct1.get_local_id(2) == 0) {
#pragma unroll
            for (int i = 0; i < TOP_K; i++) {
                assignments[token_idx * TOP_K + i] = gating::unassigned;
                offsets[token_idx * TOP_K + i] = gating::unassigned;
            }
        }
        return;
    }

    const T* token_logits = logits + token_idx * n_experts;

    float logit_val;
    if (expert_idx < n_experts) {
        logit_val = conversion::to<float>(token_logits[expert_idx]);
    } else {
        reduce::init<ROp::Max>(&logit_val);
    }
    float reduce_val = logit_val;

    int32_t local_assigned_experts[TOP_K];
    float local_assigned_logits[TOP_K];

    // Training code tends to use ``torch.argmax`` to select the expert, which
    // which has ties broken by the lower index. Since our fused comparison algorithm
    // breaks ties by the higher index (since it's the lower 32-bits of the 64-bit
    // comparison), we invert the expert index to break ties by the lower index.
    int32_t inverted_expert = n_experts - expert_idx - 1;

    // Find the top k logits
    for (int i = 0; i < TOP_K; ++i) {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, reduce_val, inverted_expert);
        local_assigned_experts[i] = n_experts - res.idx - 1;
        local_assigned_logits[i] = res.val;

        // Set the max logit to -inf so that it is not selected again
        if (item_ct1.get_local_id(2) == n_experts - res.idx - 1) {
            reduce::init<ROp::Max>(&reduce_val);
        }
    }

    const float max_logit = local_assigned_logits[0];
    float softmax_sum = sycl::native::exp(logit_val - max_logit);
    reduce::block<ROp::Add>(tb, warp, softmax_sum);

    for (int i = 0; i < TOP_K; ++i) {
        const float softmax = sycl::native::exp(local_assigned_logits[i] - max_logit) / softmax_sum;

        if (item_ct1.get_local_id(2) == 0) {
            scores[token_idx * TOP_K + i] = softmax;
            assignments[token_idx * TOP_K + i] = local_assigned_experts[i];
            offsets[token_idx * TOP_K + i] =
                dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                    expert_counts + local_assigned_experts[i], 1);
        }
    }
}

template <typename T>
void launch_top_k_gating(int32_t* expert_counts,
                         float* scores,
                         int32_t* assignments,
                         int32_t* offsets,
                         const T* logits,
                         const RaggedBatchDescriptor* batch_metadata,
                         const int32_t n_tokens,
                         const int32_t n_experts,
                         const int32_t n_top_k,
                         dpct::queue_ptr stream)
{
    const sycl::range<3> grid(1, 1, n_tokens);
    const sycl::range<3> block(
        1, 1, ((n_experts + hw_warp_size - 1) / hw_warp_size) * hw_warp_size);

    /*
    DPCT1038:13: When the kernel function name is used as a macro argument, the migration result may
    be incorrect. You need to verify the definition of the macro.
    */
    TOP_K_SWITCH(n_top_k, [&] {
        /*
        DPCT1049:14: The work-group size passed to the SYCL kernel may exceed the limit. To get the
        device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
        */
  dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});

  stream->parallel_for(
      sycl::nd_range<3>(grid * block, block),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
       top_k_gating_kernel<T, CONST_TOP_K>(
           expert_counts, scores, assignments, offsets, logits, batch_metadata, n_experts);
      });
    });
}

#define INSTANTIATE_top_k_KERNEL(T)                                                   \
    template void launch_top_k_gating<T>(int32_t * expert_counts,                     \
                                         float* scores,                               \
                                         int32_t* assignments,                        \
                                         int32_t* offsets,                            \
                                         const T* logits,                             \
                                         const RaggedBatchDescriptor* batch_metadata, \
                                         const int32_t n_tokens,                      \
                                         const int32_t n_experts,                     \
                                         const int32_t n_top_k,                       \
                                         dpct::queue_ptr stream);

INSTANTIATE_top_k_KERNEL(float) INSTANTIATE_top_k_KERNEL(sycl::half)
#ifdef BF16_AVAILABLE
    INSTANTIATE_top_k_KERNEL(sycl::ext::oneapi::bfloat16)
#endif
