// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// #include <c10/cuda/CUDAStream.h>
#include <ipex.h>
#include <torch/extension.h>
#include "activation_type.h"
#include "ds_kernel_utils.h"

template <typename T>
void launch_gated_activation(T* output,
                             const T* vals,
                             const T* bias,
                             int rows,
                             int cols,
                             ActivationType activation_type,
                             dpct::queue_ptr stream);

void ds_gated_activation(at::Tensor& output,
                         at::Tensor& input,
                         c10::optional<torch::Tensor>& bias,
                         int activation_type_raw);
