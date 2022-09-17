#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath> // for std::fabs and std::signbit

#include "glog/logging.h"
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {
    template <typename Dtype>
    void caffe_cpu_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

    template <typename Dtype>
    void caffe_cpu_axpby(const int N, const Dtype alpha, const Dtype* X
                         const Dtype beta, Dtype* Y);
    
    template <typename Dtype>
    void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

    template <typename Dtype>
    void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
                         const Dtype beta, Dtype* Y);
}