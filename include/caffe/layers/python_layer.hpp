#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;

namespace caffe {

template <typename Dtype>
class PythonLayer: public Layer<Dtype> { // 继承自Layer类
    public:
        PythonLayer(PyObject* self, const LayerParameter& param)
            : Layer<Dtype>(param), self_(bp::handle<>(bp::borrowed(self))) {}
            

    private:
        bp::object self_;
}
}