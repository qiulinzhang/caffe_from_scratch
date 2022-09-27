// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include<string>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    /// @brief Fills a Blob with constant or randomly-generated data.
    template <typename Dtype>
    class Filler {
        public:
            explicit Filler(const FillerParameter& param): filler_param_(param) {}
            // 虚函数，当将子类地址赋值给父类时，调用虚函数将调用的是子类的函数，而其他函数将是调用的是父类的函数
            virtual ~Filler() {}
            // 纯虚函数，因为本身没有实现，因此无法实例化，只能作为基类，要求子类必须实例化
            // =0 不是返回值，而是一个标志，告诉编译器这是一个纯虚函数
            virtual void Fill(Blob<Dtype>* blob) = 0;
        protected:
            FillerParameter filler_param_; // 这个类如果没有默认的无参构造函数，那么就需要上述的初始化列表直接调用拷贝构造函数
                                           // 即使有默认的无参构造函数，也可以避免一次无参构造函数的调用，更快
    }; // class Filler

    /// @brief Fills a Blob with constant values @f$ x = 0 @f$
    template <typename Dtype>
    class ConstantFiller: public Filler<Dtype> {
        public:
            explicit ConstantFiller(Const FillerParameter& param) : Filler<Dtype>(param) {}
            virtual void Fill(Blob<Dtype>* Blob) {
                Dtype* data = blob->mutable_cpu_data();
                const int count = blob->count();
                const Dtype value = this->filler_param_.value();
                CHECK(count);
                for(int i=0; i<count; ++i) {
                    data[i] = value;
                }
                CHECK_EQ(this->filler_param_.sparse(), -1) << "Sparsity not supported by this Filler.";
            }
    };

    /// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
    template <typename Dtype>
    class UniformFiller: public Filler<Dtype> {
        public:
            explicit UniformFiller(const FillerParameter& param) : Filler<Dtype>(param) {}
            virtual void Fill(Blob<Dtype>* blob) {
                CHECK(blob->count());
                caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()), Dtype(this->filler_param_.max(),
                                         blob->mutable_cpu_data()));
                CHECK_EQ(this->filler_param_.sparse(), -1) << "Sparsity not supported by this Filler.";
            }
    }
}
