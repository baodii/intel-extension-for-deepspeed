// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ds_kernel_utils.h"
#include "embed.dp.hpp"
#include "memory_access_utils.h"
#include "ragged_dtypes.h"

namespace embed {

constexpr int granularity = 16;
constexpr int threads = 512;

}  // namespace embed

template <typename TokenType, typename EmbedType>
class ragged_embed_kernel {
private:
    EmbedType* embedded_tokens;
    const TokenType* input_ids;
    const EmbedType* embedding_weight;
    const EmbedType* position_weight;
    const BatchWrapperCPP batch_desc;
    const int32_t embed_dim;
    const int32_t vocab_size;
    const int32_t max_position_embed_idx;
    const int32_t position_embed_offset;

public:
    ragged_embed_kernel(EmbedType* embedded_tokens_,
                        const TokenType* input_ids_,
                        const EmbedType* embedding_weight_,
                        const EmbedType* position_weight_,
                        const BatchWrapperCPP batch_desc_,
                        const int32_t embed_dim_,
                        const int32_t vocab_size_,
                        const int32_t max_position_embed_idx_,
                        const int32_t position_embed_offset_) : embedded_tokens(embedded_tokens_), 
                                                          input_ids(input_ids_),
                                                          embedding_weight(embedding_weight_),
                                                          position_weight(position_weight_),
                                                          batch_desc(batch_desc_),
                                                          embed_dim(embed_dim_),
                                                          vocab_size(vocab_size_),
                                                          max_position_embed_idx(max_position_embed_idx_),
                                                          position_embed_offset(position_embed_offset_) {};
    void operator()(sycl::nd_item<3>) const
    {
        
        auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
        constexpr int T_vector = embed::granularity / sizeof(EmbedType);
    
        const int32_t token_idx = item_ct1.get_group(1);
    
        // It's possible our batch is padded (under CG conditions typically)
        if (token_idx >= batch_desc.batch_metadata->n_tokens) return;
    
        TokenType token_value = input_ids[token_idx];
    
        if (token_value >= vocab_size || token_value < 0) {
            // TODO(cmikeh2): This is invalid, but not sure how we want to handle it being invalid
            // yet.
            return;
        }
    
        const EmbedType* embedding_row = embedding_weight + token_value * embed_dim;
        EmbedType* dest_row = embedded_tokens + token_idx * embed_dim;
    
        const int channel_offset =
            (item_ct1.get_local_id(2) + embed::threads * item_ct1.get_group(2)) * T_vector;
    
        if (channel_offset < embed_dim) {
            EmbedType reg_buf[T_vector];
    
            mem_access::load_global<embed::granularity>(reg_buf, embedding_row + channel_offset);
    
            if (position_weight != nullptr) {
                // Map the token to its global idx (indirect memory accesses aren't great but whatever)
                const int32_t seq_idx = batch_desc.tokens_to_seq[token_idx];
                const InflightSeqDescriptor seq_desc = batch_desc.seq_metadata[seq_idx];
                int32_t pos_emb_idx = seq_desc.seen_tokens + (token_idx - seq_desc.start_idx);
    
                // Position embed offset is an OPT-specific feature I think?
                pos_emb_idx = pos_emb_idx + position_embed_offset;
    
                // This clamping is technically
                pos_emb_idx = (pos_emb_idx < 0) ? 0 : pos_emb_idx;
                pos_emb_idx = (pos_emb_idx >= max_position_embed_idx) ? max_position_embed_idx
                                                                      : pos_emb_idx;
    
                const EmbedType* position_embedding_row = position_weight + pos_emb_idx * embed_dim;
    
                EmbedType pos_buf[T_vector];
                mem_access::load_global<embed::granularity>(pos_buf,
                                                            position_embedding_row + channel_offset);
    
#pragma unroll
                for (int i = 0; i < T_vector; i++) { reg_buf[i] += pos_buf[i]; }
            }

            mem_access::store_global<embed::granularity>(dest_row + channel_offset, reg_buf);
        }
    }

    
};

template <typename TokenType, typename EmbedType>
void launch_ragged_embed_kernel(EmbedType* embedded_tokens,
                                const TokenType* input_ids,
                                const EmbedType* embedding_weight,
                                const EmbedType* position_weight,
                                const BatchWrapperCPP batch_desc,
                                const int32_t n_tokens,
                                const int32_t embed_dim,
                                const int32_t vocab_size,
                                const int32_t max_position_embed_idx,
                                const int32_t position_embed_offset,
                                dpct::queue_ptr stream)
{
    constexpr int T_vector = embed::granularity / sizeof(EmbedType);
    constexpr int elems_per_block = embed::threads * T_vector;
    const int parallel_blocks = (embed_dim + elems_per_block - 1) / elems_per_block;

    const sycl::range<3> grid_dim(1, n_tokens, parallel_blocks);
    const sycl::range<3> block_dim(1, 1, embed::threads);

    /*
    DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

        auto capture = ragged_embed_kernel(embedded_tokens,
                                           input_ids,
                                           embedding_weight,
                                           position_weight,
                                           batch_desc,
                                           embed_dim,
                                           vocab_size,
                                           max_position_embed_idx,
                                           position_embed_offset);
        
        stream->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), capture);
    }
}

#define INSTANTIATE_EMBED_FOR_TYPES(TOKEN_TYPE, EMBED_TYPE)           \
    template void launch_ragged_embed_kernel<TOKEN_TYPE, EMBED_TYPE>( \
        EMBED_TYPE * embedded_tokens,                                 \
        const TOKEN_TYPE* input_ids,                                  \
        const EMBED_TYPE* embedding_weight,                           \
        const EMBED_TYPE* position_weight,                            \
        const BatchWrapperCPP batch_descriptor,                       \
        const int32_t n_tokens,                                       \
        const int32_t embed_dim,                                      \
        const int32_t vocab_size,                                     \
        const int32_t max_position_embed_idx,                         \
        const int32_t position_embed_offset,                          \
        dpct::queue_ptr stream);

INSTANTIATE_EMBED_FOR_TYPES(int32_t, float)
INSTANTIATE_EMBED_FOR_TYPES(int64_t, float)

INSTANTIATE_EMBED_FOR_TYPES(int32_t, sycl::half)
INSTANTIATE_EMBED_FOR_TYPES(int64_t, sycl::half)

#ifdef BF16_AVAILABLE
INSTANTIATE_EMBED_FOR_TYPES(int32_t, sycl::ext::oneapi::bfloat16)
INSTANTIATE_EMBED_FOR_TYPES(int64_t, sycl::ext::oneapi::bfloat16)
#endif
