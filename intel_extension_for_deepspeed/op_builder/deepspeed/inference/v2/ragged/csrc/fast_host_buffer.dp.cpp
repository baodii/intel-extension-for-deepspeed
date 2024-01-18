// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ds_kernel_utils.h"
#include "fast_host_buffer.h"

void* get_cuda_fast_buffer(int64_t size)
{
    void* buffer_ptr;
    // Host allocation flags that should minimize the host -> accelerator copy latency
    unsigned int alloc_flags =
        /*
        DPCT1048:0: The original value cudaHostAllocPortable is not meaningful in the migrated code
        and was removed or replaced with 0. You may need to check the migrated code.
        */
        /*
        DPCT1048:1: The original value cudaHostAllocMapped is not meaningful in the migrated code
        and was removed or replaced with 0. You may need to check the migrated code.
        */
        /*
        DPCT1048:2: The original value cudaHostAllocWriteCombined is not meaningful in the migrated
        code and was removed or replaced with 0. You may need to check the migrated code.
        */
        0 | 0 | 0;

    buffer_ptr = (void*)sycl::malloc_host(size, dpct::get_in_order_queue());
    return buffer_ptr;
}
