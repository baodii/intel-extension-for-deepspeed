# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class RaggedOpsBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_RAGGED_DEVICE_OPS"
    NAME = "ragged_device_ops"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.inference.v2.kernels.ragged_ops.{self.NAME}'

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)

    def filter_ccs(self, ccs):
        ccs_retained = []
        ccs_pruned = []
        for cc in ccs:
            if int(cc[0]) >= 8:
                # Blocked flash has a dependency on Ampere + newer
                ccs_retained.append(cc)
            else:
                ccs_pruned.append(cc)
        if len(ccs_pruned) > 0:
            self.warning(f"Filtered compute capabilities {ccs_pruned}")
        return ccs_retained

    def sources(self):
        sources = [
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/ragged_ops.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/atom_builder/atom_builder.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/blocked_flash/blocked_flash.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/embed/embed.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/embed/embed.dp.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/linear_blocked_kv_rotary/blocked_kv_rotary.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/linear_blocked_kv_rotary/blocked_kv_rotary.dp.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/logits_gather/logits_gather.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/logits_gather/logits_gather.dp.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/moe_scatter/moe_scatter.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/moe_scatter/moe_scatter.dp.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/moe_gather/moe_gather.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/moe_gather/moe_gather.dp.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/ragged_helpers/ragged_kernel_helpers.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/top_k_gating/top_k_gating.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/kernels/ragged_ops/top_k_gating/top_k_gating.dp.cpp"),
        ]

        return sources

    def include_paths(self):
        sources = [
            sycl_kernel_include('deepspeed/inference/v2/kernels/includes'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/ragged_ops'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/ragged_ops/atom_builder'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/ragged_ops/blocked_flash'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/ragged_ops/embed'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/ragged_ops/includes'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/ragged_ops/linear_blocked_kv_rotary'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/ragged_ops/logits_gather'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/ragged_ops/moe_gather'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/ragged_ops/moe_scatter'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/ragged_ops/ragged_helpers'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/ragged_ops/top_k_gating'),
            sycl_kernel_include('csrc/includes'),
        ]

        return sources
