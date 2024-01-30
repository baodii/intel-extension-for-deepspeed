# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class RaggedUtilsBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_RAGGED_OPS"
    NAME = "ragged_ops"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.inference.v2.{self.NAME}'

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
            sycl_kernel_path("deepspeed/inference/v2/ragged/csrc/fast_host_buffer.dp.cpp"),
            sycl_kernel_path("deepspeed/inference/v2/ragged/csrc/ragged_ops.cpp"),
        ]

        return sources

    def include_paths(self):
        include_dirs = [
            sycl_kernel_include('deepspeed/inference/v2/ragged/includes'),
            sycl_kernel_include('deepspeed/inference/v2/kernels/includes'),
            sycl_kernel_include('csrc/includes'),
        ]

        return include_dirs
