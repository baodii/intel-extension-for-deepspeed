// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <dpct/blas_utils.hpp>

#ifdef BF16_AVAILABLE
#endif
#ifndef __HIP_PLATFORM_HCC__
#endif
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <dpct/lib_common_utils.hpp>

class BlasContext {
    /*
    Slim wrapper for managing the lifetime of the platform's BLAS handle. This should
    be hipified for ROCm.
    */
public:
    BlasContext()
     try {
        if (DPCT_CHECK_ERROR(_handle = &dpct::get_in_order_queue()) != 0) {
            auto message = std::string("Fail to create cublas handle.");
            std::cerr << message << std::endl;
            throw std::runtime_error(message);
        }
#ifndef __HIP_PLATFORM_HCC__
        /*
        DPCT1026:0: The call to cublasSetMathMode was removed because this functionality is
        redundant in SYCL.
        */
#endif
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                << std::endl;
      std::exit(1);
    }

    virtual ~BlasContext() { _handle = nullptr; }

    static BlasContext& getInstance()
    {
        // Should always access the singleton through this function.
        static BlasContext _instance;
        return _instance;
    }

    dpct::queue_ptr get_handle() const { return _handle; }

private:
    dpct::queue_ptr _handle;
};

enum class BlasType { FP32, FP16, BF16 };

#ifdef __HIP_PLATFORM_HCC__
rocblas_operation get_trans_op(bool do_trans)
{
    return (do_trans) ? rocblas_operation_transpose : rocblas_operation_none;
}

rocblas_datatype get_datatype(BlasType type)
{
    switch (type) {
        case BlasType::FP32: return rocblas_datatype_f32_r;
        case BlasType::FP16: return rocblas_datatype_f16_r;
        case BlasType::BF16: return rocblas_datatype_bf16_r;
        default: throw std::runtime_error("Unsupported BlasType");
    }
}
#else
oneapi::mkl::transpose get_trans_op(bool do_trans) {
    return (do_trans) ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
}

dpct::library_data_t get_datatype(BlasType type)
{
    switch (type) {
        case BlasType::FP32: return dpct::library_data_t::real_float;
        case BlasType::FP16: return dpct::library_data_t::real_half;
        case BlasType::BF16: return dpct::library_data_t::real_bfloat16;
        default: throw std::runtime_error("Unsupported BlasType");
    }
}
#endif

int blas_gemm_ex(void* C,
                 const void* A,
                 const void* B,
                 int m,
                 int n,
                 int k,
                 int lda,
                 int ldb,
                 int ldc,
                 bool transa,
                 bool transb,
                 const float* alpha,
                 const float* beta,
                 BlasType type)
 try {
#ifdef __HIP_PLATFORM_HCC__
    rocblas_operation_t transa_op = get_trans_op(transa);
    rocblas_operation_t transb_op = get_trans_op(transb);

    rocblas_datatype_t abc_type = get_datatype(type);

    rocblas_status status = rocblas_gemm_ex(BlasContext::getInstance().get_handle(),
                                            transa_op,
                                            transb_op,
                                            m,
                                            n,
                                            k,
                                            (const void*)alpha,
                                            A,
                                            abc_type,
                                            lda,
                                            B,
                                            abc_type,
                                            ldb,
                                            (const void*)beta,
                                            C,
                                            abc_type,
                                            ldc,
                                            C,
                                            abc_type,
                                            ldc,
                                            rocblas_datatype_f32_r,
                                            rocblas_gemm_algo_standard,
                                            0,
                                            0);
#else
    oneapi::mkl::transpose transa_op = get_trans_op(transa);
    oneapi::mkl::transpose transb_op = get_trans_op(transb);

    dpct::library_data_t abc_type = get_datatype(type);
    int status = DPCT_CHECK_ERROR(dpct::gemm(*(BlasContext::getInstance().get_handle()),
                                             transa_op,
                                             transb_op,
                                             m,
                                             n,
                                             k,
                                             (const void*)alpha,
                                             A,
                                             abc_type,
                                             lda,
                                             B,
                                             abc_type,
                                             ldb,
                                             (const void*)beta,
                                             C,
                                             abc_type,
                                             ldc,
                                             dpct::library_data_t::real_float));
#endif

#ifdef __HIP_PLATFORM_HCC__
    if (status != rocblas_status_success) {
#else
    if (status != 0) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
            << std::endl;
  std::exit(1);
}

int blas_strided_batched_gemm(void* C,
                              const void* A,
                              const void* B,
                              int m,
                              int n,
                              int k,
                              int lda,
                              int ldb,
                              int ldc,
                              bool transa,
                              bool transb,
                              const float* alpha,
                              const float* beta,
                              int stride_A,
                              int stride_B,
                              int stride_C,
                              int batch,
                              BlasType type)
 try {
#ifdef __HIP_PLATFORM_HCC__
    rocblas_operation_t transa_op = get_trans_op(transa);
    rocblas_operation_t transb_op = get_trans_op(transb);

    rocblas_datatype_t abc_type = get_datatype(type);

    rocblas_status status =
        rocblas_gemm_strided_batched_ex(BlasContext::getInstance()::get_handle(),
                                        transa_op,
                                        transb_op,
                                        m,
                                        n,
                                        k,
                                        (const void*)alpha,
                                        A,
                                        abc_type,
                                        lda,
                                        stride_A,
                                        B,
                                        abc_type,
                                        ldb,
                                        stride_B,
                                        (const void*)beta,
                                        C,
                                        abc_type,
                                        ldc,
                                        stride_C,
                                        C,
                                        abc_type,
                                        ldc,
                                        stride_C,
                                        batch,
                                        rocblas_datatype_f32_r,
                                        rocblas_gemm_algo_standard,
                                        0,
                                        0);
#else
    oneapi::mkl::transpose transa_op = get_trans_op(transa);
    oneapi::mkl::transpose transb_op = get_trans_op(transb);

    dpct::library_data_t abc_type = get_datatype(type);

    int status = DPCT_CHECK_ERROR(dpct::gemm_batch(*(BlasContext::getInstance().get_handle()),
                                                   transa_op,
                                                   transb_op,
                                                   m,
                                                   n,
                                                   k,
                                                   (const void*)alpha,
                                                   A,
                                                   abc_type,
                                                   lda,
                                                   stride_A,
                                                   B,
                                                   abc_type,
                                                   ldb,
                                                   stride_B,
                                                   (const void*)beta,
                                                   C,
                                                   abc_type,
                                                   ldc,
                                                   stride_C,
                                                   batch,
                                                   dpct::library_data_t::real_float));
#endif

#ifdef __HIP_PLATFORM_HCC__
    if (status != rocblas_status_success) {
#else
    if (status != 0) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (batch: %d, m: %d, n: %d, k: %d, error: %d) \n",
                batch,
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
            << std::endl;
  std::exit(1);
}
