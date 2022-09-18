#include "caffe/common.cpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

    template<>
    void caffe_scal<float>(const int N, const float alpha, float* X) {
        cblas_sscal(N, alpha, X, 1);
    }

    template<>
    void caffe_scale<double>(const int N, const double alpha, double* X) {
        cblas_dscal(N, alpha, X, 1);
    }

    template <>
    void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                                const float beta, float* Y) {
        cblas_saxpby(N, alpha, X, 1, beta, Y, 1);          
    }

    template<>
    void caffe_cpu_axpby<double> (const int N, const double alpha, const double* X,
                                  const double beta, double* Y) {
        cblas_dxpby(N, alpha, X, 1, beta, Y, 1);
    }

    template<>
    void caffe_add<float>(const int n, const float* a, const float* b, float* y) {
        vsAdd(n, a, b, y);
    }

    template<>
    void caffe_add(double)(const int n, const double* a, const double* b, double* y) {
        vdAdd(n, a, b, y);
    }

    template <>
    void caffe_sub<float>(const int n, const float* a, const float* b,
        float* y) {
      vsSub(n, a, b, y);
    }

    template <>
    void caffe_sub<double>(const int n, const double* a, const double* b,
        double* y) {
      vdSub(n, a, b, y);
    }

    template <>
    void caffe_mul<float>(const int n, const float* a, const float* b,
        float* y) {
      vsMul(n, a, b, y);
    }

    template <>
    void caffe_mul<double>(const int n, const double* a, const double* b,
        double* y) {
      vdMul(n, a, b, y);
    }

    template <>
    void caffe_div<float>(const int n, const float* a, const float* b,
        float* y) {
      vsDiv(n, a, b, y);
    }

    template <>
    void caffe_div<double>(const int n, const double* a, const double* b,
        double* y) {
      vdDiv(n, a, b, y);
        }

    template<>
    void caffe_sqr<float>(const int n, const float* a, float* y) {
        vsSqr(n, a, y);
    }

    template <>
    void caffe_powx<float>(const int n, const float* a, const float b,
        float* y) {
      vsPowx(n, a, b, y);
    }

    template <>
    void caffe_powx<double>(const int n, const double* a, const double b,
        double* y) {
      vdPowx(n, a, b, y);
    }

    template<>
    void caffe_sqr<double>(const int n, const double* a, double* y) {
        vdSqr(n, a, y);
    }

    template<>
    void caffe_sqrt<float>(const int n, const float* a, float* y) {
        vsSqrt(n, a, y);
    }

    template<>
    void caffe_sqrt<double>(const int n, const double* a, double* y) {
        vdSqrt(n, a, y);
    }

    template<>
    void caffe_exp<float>(const int n, const float* a, float* y) {
        vsExp(n, a, y);
    }

    template<>
    void caffe_exp<double>(const int n, const double* a, double* y) {
        vdExp(n, a, y);
    }

    template <>
    void caffe_log<float>(const int n, const float* a, float* y) {
        vsLn(n, a, y);
    }

    template <>
    void caffe_log<double>(const int n, const double* a, double* y) {
        vdLn(n, a, y);
    }

    template <>
    void caffe_abs<float>(const int n, const float* a, float* y) {
        vsAbs(n, a, y);
    }
    
    template <>
    void caffe_abs<double>(const int n, const double* a, double* y) {
        vdAbs(n, a, y);
    }

    template<>
    void caffe_sqr<float>(const int n, const float* a, float* y) {
        vsSqr(n, a, y);
    }

    template<>
    void caffe_sqr<double>(const int n, const double* a, double* y) {
        vdSqr(n, a, y);
    }

    template<>
    void caffe_sqrt<float>(const int n, const float* a, float* y) {
        vsSqrt(n, a, y);
    }

    template<>
    void caffe_sqrt<double>(const int n, const double* a, double* y) {
        vdSqrt(n, a, y);
    }

    template<>
    void caffe_exp<float>(const int n, const float* a, float* y) {
        vsExp(n, a, y);
    }

    template<>
    void caffe_exp<double>(const int n, const double* a, double* y) {
        vdExp(n, a, y);
    }

    template <>
    void caffe_log<float>(const int n, const float* a, float* y) {
        vsLn(n, a, y);
    }

    template <>
    void caffe_log<double>(const int n, const double* a, double* y) {
        vdLn(n, a, y);
    }

    template <>
    void caffe_abs<float>(const int n, const float* a, float* y) {
        vsAbs(n, a, y);
    }
    
    template <>
    void caffe_abs<double>(const int n, const double* a, double* y) {
        vdAbs(n, a, y);
    }
}