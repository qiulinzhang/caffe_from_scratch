#ifndef CAFFE_UTIL_MKL_ALTERNATE_H_
#define CAFFE_UTIL_MKL_ALTERNATE_H_

#ifdef USE_MKL

#include <mkl.h> // intel 开发的mkl库，基础的数学运算

#else
// 定义一些如果没有 mkl.h 这个库，手动实现的一些需要用到的 mkl.h 里面的某些函数，
// 即文件名所指出的 alternate，替代 备份，通过 math.h 中的库来手动实现
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
// Functions that caffe uses but are not present if MKL is not linked.

// A simple way to define the vsl unary functions. The operation should
// be in the form e.g. y[i] = sqrt(a[i])

#define DEFINE_VSL_UNARY_FUNC(name, operatrion) \
  template<typename Dtype> \ 
  void v##name(const int n, const Dtype* a, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int i=0; i < n; ++i) {operation;}; \
  } \
  inline void vs##name(const int n, const float* a, float* y) { \
    v##name<float>(n, a, y); \
  } \
  inline void vd##name(const int n, const double* a, double* y) { \
    v##name<double>(n, a, y); \
  }

DEFINE_VSL_UNARY_FUNC(Sqr, y[i]=a[i]*a[i])
DEFINE_VSL_UNARY_FUNC(Sqrt, y[i]=sqrt(a[i]))
DEFINE_VSL_UNARY_FUNC(Exp, y[i]=exp(a[i]))
DEFINE_VSL_UNARY_FUNC(Ln, y[i]=log(a[i]))
DEFINE_VSL_UNARY_FUNC(Abs, y[i]=fabs(a[i]))

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
  template <typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for(int i=0; i<n; ++i) {operation;} \
  } \
  inline void vs##name(const int n, const float* a, const float b, float* y) {\
    v##name<float>(n, a, b, y); \
  } \
  inline void vd##name(const int n, const double* a, const float b, double* y) {\
    v##name<double>(n, a, b, y); \
  }

DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = pow(a[i], b))

// A simple way to define the vsl binary functions. The operation should
// be in the form e.g. y[i] = a[i] + b[i]
#define DEFINE_VSL_BINARY_FUNC(name, operation) \
  template <typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype*b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y); \
    for(int i=0; i<n; ++i) { operation; } \
  } \
  inline vs##name(const int n, const float* a, const float* b, float* y) { \
    v##name<float>(n, a, b, y); \
  } \
  inline vd##name(const int n, const double* a, const double* b, double* y) { \
    v##name<double>(n, a, b, y); \
  }

DEFINE_VSL_BINARY_FUNC(Add, y[i]=a[i] + b[i])
DEFINE_VSL_BINARY_FUNC(Sub, y[i]=a[i] - b[i])
DEFINE_VSL_BINARY_FUNC(Mul, y[i]=a[i] * b[i])
DEFINE_VSL_BINARY_FUNC(Div, y[i]=a[i] / b[i])

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

#endif // USE_MKL
#endif // CAFFE_UTIL_MKL_ALTERNATE_H_