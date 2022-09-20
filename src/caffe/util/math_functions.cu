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
        int lda = (TransA == CblasNoTrans) ? K : M;
        int ldb = (TransB == CblasNoTrans) ? N : K;
        cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
        CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransA, cuTransB, 
                     N, M, K,
                     &alpha, B, ldb, A, lda,
                     &beta, C, N));
    }
}