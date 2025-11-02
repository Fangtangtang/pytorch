#pragma once

#include "Kernel.h"

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>   // TORCH_CUDABLAS_CHECK
#include <c10/util/BFloat16.h>
#include <c10/core/ScalarType.h>
#include <iostream>
#include <type_traits>              // std::is_same_v

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
template <typename C_Dtype>
class GemmInternalCublasBF16Kernel : public Kernel {
public:
    /**
     * @brief 构造函数，捕获 cublasGemmEx 调用所需的所有状态。
     * * @param handle cublas 句柄
     * @param opa 矩阵 A 的转置操作
     * @param opb 矩阵 B 的转置操作
     * @param m 矩阵 A 和 C 的行数
     * @param n 矩阵 B 和 C 的列数
     * @param k 矩阵 A 的列数 / B 的行数
     * @param falpha 缩放因子 alpha (原函数中的局部变量，因此按值捕获)
     * @param a 指向矩阵 A 的设备指针
     * @param lda 矩阵 A 的 leading dimension
     * @param b 指向矩阵 B 的设备指针
     * @param ldb 矩阵 B 的 leading dimension
     * @param fbeta 缩放因子 beta (原函数中的局部变量，因此按值捕W获)
     * @param c 指向矩阵 C 的设备指针 (输入/输出)
     * @param ldc 矩阵 C 的 leading dimension
     * @param compute_type cublas 的计算类型
     * @param cublas_flags cublas 的数学模式标志
     */
    GemmInternalCublasBF16Kernel(
        cublasHandle_t handle,
        cublasOperation_t opa,
        cublasOperation_t opb,
        int64_t m,
        int64_t n,
        int64_t k,
        float falpha, // 按值捕获
        const BFloat16* a,
        int64_t lda,
        const BFloat16* b,
        int64_t ldb,
        float fbeta,  // 按值捕获
        C_Dtype* c,
        int64_t ldc,
        cudaDataType_t compute_type,
        cublasMath_t cublas_flags
    ) : // 使用成员初始化列表存储所有捕获的参数
        handle_(handle), opa_(opa), opb_(opb),
        m_(m), n_(n), k_(k),
        falpha_(falpha), a_(a), lda_(lda),
        b_(b), ldb_(ldb),
        fbeta_(fbeta), c_(c), ldc_(ldc),
        compute_type_(compute_type),
        cublas_flags_(cublas_flags)
    {
        // 构造函数体
        // 
        // 我们可以在这里打印 "已入队" 的消息，
        // 这将在调用 enqueue() 的线程 (PyTorch 线程) 中立即执行。
        std::cout << "GemmInternalCublasBF16Kernel: [已入队] m=" << m_ << ", n=" << n_ << ", k=" << k_ << std::endl;
    }

    /**
     * @brief 执行被封装的 cublasGemmEx 调用。
     * * 此方法由 KernelManager 的工作线程调用。
     * 这里的代码基本上就是从 `gemm_internal_cublas_bfloat16_helper` 中
     * 移动过来的原始 BLAS 调用逻辑。
     */
    void execute() override {
        // 1. 设置数学模式 (从原函数移动而来)
        TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle_, cublas_flags_));

        // 2. 打印 "正在执行" 的消息 (原函数中的 printf hook)
        //    这将在工作线程中执行。
        std::cout << "GemmInternalCublasBF16Kernel: [执行中] m=" << m_ << ", n=" << n_ << ", k=" << k_ << std::endl;

        // 3. 执行核心的 cublasGemmEx 调用
        //    注意：我们使用存储的成员变量
        TORCH_CUDABLAS_CHECK(cublasGemmEx(
            handle_,
            opa_,
            opb_,
            m_,
            n_,
            k_,
            &falpha_, // 注意：使用成员变量的地址
            a_,
            CUDA_R_16BF,
            lda_,
            b_,
            CUDA_R_16BF,
            ldb_,
            &fbeta_,  // 注意：使用成员变量的地址
            c_,
            // 根据模板参数 C_Dtype 动态确定输出类型
            std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16BF,
            ldc_,
            compute_type_,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // 4. 恢复默认的数学模式 (从原函数移动而来)
        TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));

        // 5. (可选) 打印完成消息
        std::cout << "GemmInternalCublasBF16Kernel: [已完成] m=" << m_ << ", n=" << n_ << ", k=" << k_ << std::endl;
    }

private:
    // --- 存储的成员变量 ---
    // 这些变量是在构造时从 `gemm_internal_cublas_bfloat16_helper` 捕获的
    cublasHandle_t handle_;
    cublasOperation_t opa_, opb_;
    int64_t m_, n_, k_;
    float falpha_; // 存储 alpha 的值
    const BFloat16* a_;
    int64_t lda_;
    const BFloat16* b_;
    int64_t ldb_;
    float fbeta_;  // 存储 beta 的值
    C_Dtype* c_;
    int64_t ldc_;
    cudaDataType_t compute_type_;
    cublasMath_t cublas_flags_;
};
