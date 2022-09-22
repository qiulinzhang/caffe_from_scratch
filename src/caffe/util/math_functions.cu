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
        // A: CPU中是MxK, 传输给GPU按照 lda=K 的列主序读，GPU得到 KxM
        // B: CPU中是KxN, 传输给GPU按照 ldb=N 的列主序读，GPU得到 NxK
        // C: CPU中是MxN，传输给GPU按照 ldc=N 的列主序读，GPU得到 NxM
        // 所以在调用 cublasSgemm 库时，需要按照 C = A * B = [(B^T) * (A^T)]^T 式子
        // 把 B 和 A 调换位置，
        // 然后行列的参数也要从 M (即 MxK, A), N (即 NxK, B), K 变成 N(即 NxK, B^T), M(即 MxK, A^T), K
        // 最后得到的shape也是 NxM, 即 C^T，直接传输给CPU按照行主序读，就能得到 MxN，即 C
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

    template<>
    void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N, 
                               const float alpha, const float* A, const float* x, 
                               const float beta, float* y) {
        // 计算 y = alpha*op(A)*x + beta*y, op() 表示根据 TransA 是否对矩阵A进行转置
        cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
        // 这里首先定义了一个转置flag，即原来没有转置，但要进行转置 CUBLAS_OP_T
        // 首先 A: MxN, GPU按照lda=N的列主序读，得到 NxM, 
        // 因此需要将cublasSgemv 的rows 和 colums 分别设置为 N, M
        // 然后根据标志位 cuTransA==CUBLAS_OP_T，对GPU读到的 NxM矩阵进行转置，得到 MxN
        // 然后与长度为N的向量 x 进行乘法，得到长度为 M 的向量 y
        CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, 
                                 &alpha, A, N, x, 1, &beta, y, 1));
    }

    template<>
    void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                                const double alpha, const double* A, const double* x,
                                const double beta, double* y) {
        cublasOperation_t cuTransA = (TransA==CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
        CBULAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M,
                                 &alpha, A, N, x, 1, &beta, y, 1));
    }

    template <>
    void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
        float* Y) {
      CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
    }

    template <>
    void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
        double* Y) {
      CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
    }

    void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
        if(X!=Y) {
            CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));
        }
    }

    template <>
    void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
      CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
    }

    template <>
    void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
      CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
    }

    tepmlate <>
    void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                               cudaStream_t str) {
        // 在某一个流 str 后执行 cublasSscal 操作, problem，这个函数似乎没有被调用过
        cudaStream_t initial_stream;
        // 获取当前的流，并保存用以后续恢复
        CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
        // 把流设置为目标流
        CUBLAS_CHECK(cublasSetStram(Caffe::cublas_handle(), str));
        // 在目标流后执行 运算
        CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
        // 再恢复当前流
        CUBLAS_CHECK(cublasSetStram(Caffe::cublas_handle(), initial_stream));
    }
template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}
template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

}