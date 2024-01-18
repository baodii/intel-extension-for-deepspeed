// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ds_kernel_utils.h"
#include "logits_gather.dp.hpp"
#include "memory_access_utils.h"
#include "ragged_dtypes.h"

namespace logits_gather {

constexpr int granularity = 16;
constexpr int threads = 512;

}  // namespace logits_gather

template <typename T>
void logits_gather_kernel(T* final_token_acts,
                                     const T* token_acts,
                                     const RaggedBatchDescriptor* ragged_batch,
                                     const InflightSeqDescriptor* inflight_batch,
                                     const int32_t embed_dim)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int T_vector = logits_gather::granularity / sizeof(T);

    const int32_t seq_id = item_ct1.get_group(1);

    // It's possible we've padded the output Tensor (under CG conditions)
    if (seq_id >= ragged_batch->n_sequences) return;

    const InflightSeqDescriptor seq = inflight_batch[seq_id];
    const int final_token_idx = seq.start_idx + seq.n_tokens - 1;

    const int token_offset = final_token_idx * embed_dim;
    const int thread_offset = item_ct1.get_local_id(2) * T_vector +
                              item_ct1.get_group(2) * logits_gather::threads * T_vector;

    const int final_token_offset = seq_id * embed_dim;

    T reg_buf[T_vector];

    if (thread_offset < embed_dim) {
        mem_access::load_global<logits_gather::granularity>(
            reg_buf, token_acts + token_offset + thread_offset);

        mem_access::store_global<logits_gather::granularity>(
            final_token_acts + final_token_offset + thread_offset, reg_buf);
    }
}

template <typename T>
void launch_logits_gather(T* final_token_acts,
                          const T* all_acts,
                          const RaggedBatchDescriptor* ragged_batch,
                          const InflightSeqDescriptor* inflight_batch,
                          const int32_t n_seqs,
                          const int32_t embed_dim,
                          dpct::queue_ptr stream)
{
    constexpr int T_vector = logits_gather::granularity / sizeof(T);
    constexpr int elems_per_block = logits_gather::threads * T_vector;
    const int parallel_blocks = (embed_dim + elems_per_block - 1) / elems_per_block;

    const sycl::range<3> grid(1, n_seqs, parallel_blocks);
    const sycl::range<3> block(1, 1, logits_gather::threads);

    /*
    DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
                logits_gather_kernel<T>(
                    final_token_acts, all_acts, ragged_batch, inflight_batch, embed_dim);
            });
    }
}

#define INSTANTIATE_FOR_TYPE(T)                                                        \
    template void launch_logits_gather<T>(T * final_token_acts,                        \
                                          const T* all_acts,                           \
                                          const RaggedBatchDescriptor* ragged_batch,   \
                                          const InflightSeqDescriptor* inflight_batch, \
                                          const int32_t n_seqs,                        \
                                          const int32_t embed_dim,                     \
                                          dpct::queue_ptr stream);

INSTANTIATE_FOR_TYPE(float)
INSTANTIATE_FOR_TYPE(sycl::half)

#ifdef BF16_AVAILABLE
INSTANTIATE_FOR_TYPE(sycl::ext::oneapi::bfloat16)
#endif
