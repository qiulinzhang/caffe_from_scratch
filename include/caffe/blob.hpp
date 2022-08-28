#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int KMaxBlobAxes = 32;

namespace caffe{

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Blob{
    public:
        Blob(): data_(), diff_(), count_(0), capacity_(0){}
    
    /// @brief Deprecated; use <code>Blob(const vector<int>& shape)<code>
    explicit Blob(const int num, const int channels, const int height, const int width);
    explicit Blob(const vectot<int>& shape); // 引用，相当于别名，不会给形参创建内存，另外const不允许对引用进行修改
    
    /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.    
    void Reshape(const int num, const int channles, const int height, const int width);
    
}
}