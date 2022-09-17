#include "caffe/common.cpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

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
}