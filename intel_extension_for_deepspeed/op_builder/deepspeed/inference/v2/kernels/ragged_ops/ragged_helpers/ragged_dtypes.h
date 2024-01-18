// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <stdint.h>

struct
#ifdef __CUDA_CC__
    __align__(8)
#endif dpct_type_131076
{
    int32_t n_tokens;
    int32_t n_sequences;
}
typedef RaggedBatchDescriptor;

struct
#ifdef __CUDA_CC__
    __align__(16)
#endif dpct_type_683363
{
    int32_t start_idx;
    int32_t n_tokens;
    int32_t seen_tokens;
    int32_t UNUSED;  // Explicit padding to match the Python code pattern.
}
typedef InflightSeqDescriptor;

struct
#ifdef __CUDA_CC__
    __align__(8)
#endif dpct_type_132419
{
    int32_t** block_lists;
    int32_t block_size;
    int32_t n_blocks;
}
typedef KVCacheDescriptor;

struct dpct_type_375691 {
    const RaggedBatchDescriptor* batch_metadata;  // Offset 0
    const InflightSeqDescriptor* seq_metadata;    // Offset 8
    const int32_t* tokens_to_seq;                 // Offset 16
    const KVCacheDescriptor kv_desc;              // Offset 24
} typedef BatchWrapperCPP;
