#include<climits>
#include<vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    void Blob<Dtype>::Reshape(const int num, const int channels, const int height, const int width) {
        vector<int> shape(4);
        shape[0] = num;
        shape[1] = channels;
        shape[2] = height;
        shape[3] = width;
        Reshape(shape);
    }   

    template <typename Dtype>
    void Blob<Dtype>::Reshape(const vector<int>& shape) {
        CHECK_LE(shape.size(), kMaxBlobAxes);
        count_ = 1;
        shape_.resize(shape.size());
        if (!shape_data_ || shape_data_->size() < shape.size()*sizeof(int)) {
            shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
        }
        int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
        for (int i=0; i<shape.size(); ++i){
            CHECK_GE(shape[i], 0);
            if (count_!=0){
                CHECK_LE(shape[i], INT_MAX/count_) << "blob size exceeds INT_MAX";
            }
            count_ *= shape[i];
            shape_[i] = shape[i];
            shape_data[i] = shape[i];
        }
        if (count_ > capacity_) {
            capacity_ = count_;
            data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
            data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
        }
    }

    template <typename Dtype>
    void Blob<Dtype>::Reshape(const BlobShape& shape) {
        CHECK_LE(shape.dim_size(), kMaxBlobAxes);
        vector<int> shape_vec(shape.dim_size());
        for(int i=0; i<shape.dim_size(); ++i) {
            shape_vec[i] = shape.dim(i);
        }
        Reshape(shape_vec)
    }

    template <typename Dtype>
    void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
        Reshape(other.shape());
    }
}