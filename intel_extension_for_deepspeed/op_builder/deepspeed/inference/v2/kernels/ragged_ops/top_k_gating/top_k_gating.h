// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// #include <c10/cuda/CUDAStream.h>
#include <ipex.h>
#include <torch/extension.h>
#include "ragged_dtypes.h"
#include "top_k_gating.dp.hpp"

/*
Perform softmax plus atomics to get token mapping.
*/
void top_k_gating(torch::Tensor& expert_counts,
                  torch::Tensor& scores,
                  torch::Tensor& assignments,
                  torch::Tensor& offsets,
                  torch::Tensor& logits,
                  torch::Tensor& batch_metadata);
