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
    /**
     * @brief Change the dimensions of the blob, allocating new memory if
     *        necessary.
     *
     * This function can be called both to create an initial allocation
     * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
     * or Layer::Forward. When changing the size of blob, memory will only be
     * reallocated if sufficient memory does not already exist, and excess memory
     * will never be freed.
     *
     * Note that reshaping an input blob and immediately calling Net::Backward is
     * an error; either Net::Forward or Net::Reshape need to be called to
     * propagate the new input shape to higher layers.
     */
    void Reshape(const vectot<int>& shape);
    void Reshape(const BlobShape& shape);
    void ReshapeLike(const Blob& other);
    inline string shape_string() const{
        ostringstream stream;
        for (int i=0; i<shape_.size(); ++i){
            stream << shape_[i]<<" ";
        }
        stream << "("<< count_ << ")";
        return stream.str();
    }
    inline const vector<int>& shape() const{return shape_;}

    /**
     * @brief Returns the dimension of the index-th axis (or the negative index-th
     *        axis from the end, if index is negative).
     *
     * @param index the axis index, which may be negative as it will be
     *        "canonicalized" using CanonicalAxisIndex.
     *        Dies on out of range index.
     */
    inline int shape(int index) const{
        return shape_[CanonicalAxisIndex(index)];
    }

    inline int num_axes() const{return shape_.size();} // vector的大小，>= capacity
    inline int count() const {return count_};
    /**
     * @brief Returns the 'canonical' version of a (usually) user-specified axis,
     *        allowing for negative indexing (e.g., -1 for the last axis).
     *
     * @param axis_index the axis index.
     *        If 0 <= index < num_axes(), return index.
     *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
     *        e.g., the last axis index (num_axes() - 1) if index == -1,
     *        the second to last if index == -2, etc.
     *        Dies on out of range index.
     */
    
    inline int CanonicalAxisIndex(int axis_index) const {
        CHECK_GE(axis_index, -num_axes()) 
            << "axis "<<axis_index<<" out of range for "<< num_axes()
            << " -D Blob with shape " << shape_string();
        CHECK_LT(axis_index, num_axes())
            << "axis "<< axis_index << " out of range for " <<  num_axes()
            << " -D Blob with shape " << shape_string();
        
        if (axis_index < 0){
            return axis_index + num_axes();
        }
        return axis_index;
    }
     
}
}