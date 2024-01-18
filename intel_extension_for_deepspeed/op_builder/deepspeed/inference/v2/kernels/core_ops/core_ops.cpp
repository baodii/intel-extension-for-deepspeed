// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// #include <c10/cuda/CUDAStream.h>
#include <ipex.h>
#include <torch/extension.h>

#include "bias_activation.h"
#include "blas.h"
#include "gated_activation_kernels.h"
#include "layer_norm.h"
#include "rms_norm.h"
#include "ds_kernel_utils.h"

namespace at {
  namespace cuda {
    dpct::queue_ptr getCurrentCUDAStream() {
      auto device_type = c10::DeviceType::XPU;
      c10::impl::VirtualGuardImpl impl(device_type);
      c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
      auto& queue = xpu::get_queue_from_stream(c10_stream);
      return &queue;
    }

    dpct::queue_ptr getStreamFromPool(bool) {
      // not implemented
      return nullptr;
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // bias_activation.h
    m.def("bias_activation", &bias_activation, "DeepSpeed bias activation in CUDA");

    // layer_norm.h
    m.def("layer_norm", &ds_layer_norm, "DeepSpeed layer norm in CUDA");
    m.def("pre_layer_norm", &ds_pre_layer_norm, "DeepSpeed pre layer norm in CUDA");
    m.def("post_layer_norm", &ds_post_layer_norm, "DeepSpeed pre layer norm in CUDA");

    // blas.h
    m.def("blas_linear", &blas_linear, "Linear implemented by vendor BLAS");
    m.def("blas_4d_matmul", &blas_4d_matmul, "4D matmul implemented by vendor BLAS");
    m.def("create_handle", &create_handle, "Create a handle for vendor BLAS");

    // gated_activation_kernels.h
    m.def("gated_activation", &ds_gated_activation, "DeepSpeed gated activation in CUDA");

    // rms_norm.h
    m.def("rms_norm", &rms_norm, "DeepSpeed rms norm in CUDA");
    m.def("rms_pre_norm", &rms_pre_norm, "DeepSpeed rms pre norm in CUDA");
}
