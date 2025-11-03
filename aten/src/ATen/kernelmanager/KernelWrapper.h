#pragma once

/**
 * 12.4: https://docs.nvidia.com/cuda/archive/12.4.0/pdf/CUBLAS_Library.pdf
 * 12.8: https://docs.nvidia.com/cuda/archive/12.8.0/pdf/CUBLAS_Library.pdf
 */

#include "Kernel.h"

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <iostream>
#include <type_traits> // std::is_same_v

// 确保 at::BFloat16 类型可用
using at::BFloat16;

/**
 * @brief GemmInternalCublasBF16Kernel
 *
 * 这是一个具体的 Kernel 实现，封装了对 cublasGemmEx 的一次特定调用。
 * 它的目的是捕获 `gemm_internal_cublas_bfloat16_helper` 函数中
 * 的所有必要参数，以便稍后在 `execute()` 方法中执行它。
 *
 * @tparam C_Dtype 矩阵 C (输出) 的数据类型。
 */
class CublasGemmExKerenl : public Kernel {
 public:
  // please refer to args in cublasGemmEx
  CublasGemmExKerenl(cublasHandle_t handle,
                     cublasMath_t cublas_flags,
                     const cublasOperation_t* opa,
                     const cublasOperation_t* opb,
                     long int m,
                     long int n,
                     long int k,
                     const void* alpha,
                     const void* Aarray,
                     cudaDataType_t Atype,
                     long int lda,
                     const void* Barray,
                     cudaDataType_t Btype,
                     long int ldb,
                     const void* beta,
                     void* Carray,
                     cudaDataType_t Ctype,
                     long int ldc,
                     cudaDataType_t computeType, // corresponding to `CUDA_R_32F` (mismatch the arg type in doc)
                     cublasGemmAlgo_t algo)
      : handle_(handle),
        cublas_flags_(cublas_flags),
        opa_(opa),
        opb_(opb),
        m_(m),
        n_(n),
        k_(k),
        alpha_(alpha),
        Aarray_(Aarray),
        Atype_(Atype),
        lda_(lda),
        Barray_(Barray),
        Btype_(Btype),
        ldb_(ldb),
        beta_(beta),
        Carray_(Carray),
        Ctype_(Ctype),
        ldc_(ldc),
        compute_type_(computeType),
        algo_(algo) {
    std::cout << "cublasGemmEx: [已入队] m=" << m_ << ", n=" << n_ << ", k=" << k_ << std::endl;
  }

  void execute() override {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle_, cublas_flags_));
    std::cout << "cublasGemmEx: [执行中] m=" << m_ << ", n=" << n_ << ", k=" << k_ << std::endl;
    TORCH_CUDABLAS_CHECK(cublasGemmEx(handle_,
                                      *opa_,
                                      *opb_,
                                      m_,
                                      n_,
                                      k_,
                                      alpha_,
                                      Aarray_,
                                      Atype_,
                                      lda_,
                                      Barray_,
                                      Btype_,
                                      ldb_,
                                      beta_,
                                      Carray_,
                                      Ctype_,
                                      ldc_,
                                      compute_type_,
                                      algo_));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
    std::cout << "cublasGemmEx: [已完成] m=" << m_ << ", n=" << n_ << ", k=" << k_ << std::endl;
  }

 private:
  cublasHandle_t handle_;
  cublasMath_t cublas_flags_;
  const cublasOperation_t* opa_;
  const cublasOperation_t* opb_;
  long int m_, n_, k_;
  const void* alpha_;
  const void* Aarray_;
  cudaDataType_t Atype_;
  long int lda_;
  const void* Barray_;
  cudaDataType_t Btype_;
  long int ldb_;
  const void* beta_;
  void* Carray_;
  cudaDataType_t Ctype_;
  long int ldc_;
  cudaDataType_t compute_type_;
  cublasGemmAlgo_t algo_;
};
