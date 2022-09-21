#include <math_functions.h> // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
    template <>
    void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                               const int M, const int N, const int K,
                               const float alpha, const float* A, const float* B,
                               const float beta, float* C) {
        // Note that cublas follows fortran order
        // intel cpu 是行主序，二维数组在在内存中是按行连续的，即 MxK的二维数组，是每K个进行顺序存储的
        // cuda 采用的是列主序，二维数组在显存中是按列存储的
        // 只要leading edge 保持不变，cpu中行主序的数据按照列主序来读的话就是其转置
        // 因此 leading edge的设置与cpu的设置相同
        int lda = (TransA == CblasNoTrans) ? K : M; 
        int ldb = (TransB == CblasNoTrans) ? N : K;
        // C = A * B = [(B^T) * (A^T)]^T
        cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
        CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransA, cuTransB, 
                     N, M, K,
                     &alpha, B, ldb, A, lda,
                     &beta, C, N));
        // 注意上面相比较于CPU的方式，调换了 B 和 A 的顺序，也就是在显卡中计算的是 (B^T) * (A^T)
        // 即 得到的是 C^T 存储在 C 中，然后再将C直接传给CPU即可，
        // 这是因为，是要保持leading edge 不变，显卡中 列主序 的C^T 按照CPU的行主序来读的话就是 C
    }

    template<>
    void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
                        const int M, const int N, const int K, 
                        const double alpha, const double* A, const double* B,
                        const double beta, double* C) {
        // Note that cublas follows fortran order
        int lda = (TransA == CblasNoTrans) ? K : M;
        int ldb = (TransB == CblasNoTrans) ? N : K;
        cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N: CUBLAS_OP_T;
        cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N: CUBLAS_OP_T;
        CBULAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransA, cuTransB, 
                                 N, M, K,
                                 &alpha, B, ldb, A, lda,
                                 &beta, C, N));
    }
}