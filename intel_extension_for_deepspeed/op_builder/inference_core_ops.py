# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class InferenceCoreBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_INFERENCE_CORE_OPS"
    NAME = "inference_core_ops"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.inference.v2.kernels{self.NAME}'

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)

    def filter_ccs(self, ccs):
        ccs_retained = []
        ccs_pruned = []
        for cc in ccs:
            if int(cc[0]) >= 6:
                ccs_retained.append(cc)
            else:
                ccs_pruned.append(cc)
        if len(ccs_pruned) > 0:
            self.warning(f"Filtered compute capabilities {ccs_pruned}")
        return ccs_retained

    def sources(self):
        sources = [
            sycl_kernel_path("deepspeed/inference/v2/kernels/core_ops/core_ops.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation.dp.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/core_ops/cuda_layer_norm/layer_norm.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/core_ops/cuda_layer_norm/layer_norm.dp.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/core_ops/cuda_rms_norm/rms_norm.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/core_ops/cuda_rms_norm/rms_norm.dp.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/core_ops/gated_activations/gated_activation_kernels.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/core_ops/gated_activations/gated_activation_kernels.dp.cpp"),
        ]

        return sources

    def include_paths(self):
        sources = [
            sycl_kernel_include('deepspeed/inference/v2/kernels/core_ops/bias_activations'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/core_ops/blas_kernels'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/core_ops/cuda_layer_norm'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/core_ops/cuda_rms_norm'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/core_ops/gated_activations'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/includes'),
            sycl_kernel_include('csrc/includes'),
        ]

        return sources
