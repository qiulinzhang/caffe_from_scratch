#ifndef CAFFE_UTIL_MKL_ALTERNATE_H_
#define CAFFE_UTIL_MKL_ALTERNATE_H_

#ifdef USE_MKL

#include <mkl.h> // intel 开发的mkl库，基础的数学运算

#else

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
extern "C" {
    #include<cblas.h>
} // extern "C" 是指采用C规范来链接<cblas.h>，原因是，C++语言在编译的时候为了解决函数的多态问题，
  // 会将函数名和参数联合起来生成一个中间的函数名称，
  // 而C语言则不会，因此会造成链接时无法找到对应函数的情况，此时C函数就需要用extern
#endif // USE_ACCELERATE

#include <math.h>

// In addition, MKL comes with an additional function axpby that is not present
// in standard blas. We will simply use a two-step (inefficient, of course) way
// to mimic that.
// MKL库中包含了标准blas库中没有的扩展库，比如 z=a*x+b*y
// 但是可以通过 k=a*x 和 z = z+b*y 两步标准blas库实现
inline void cblas_saxpby(const int N,  
                         const float alpha, const float* X, const int incX, 
                         const float beta, const float* Y, const int incY){
        cblas_sscal(N, beta, Y, incY);
        cblas_saxpy(N,  alpha, X, incX, Y, incY);
        }

inline void cblas_daxpby(const int N, const double alpha, const double* X,
                         const int incX, const double beta, double* Y,
                         const int incY) {
  cblas_dscal(N, beta, Y, incY);
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}