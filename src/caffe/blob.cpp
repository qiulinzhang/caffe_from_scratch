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
        // problem 没有对里面的数据进行reshape??
        CHECK_LE(shape.size(), kMaxBlobAxes);
        count_ = 1;
        shape_.resize(shape.size());
        if (!shape_data_ || shape_data_->size() < shape.size()*sizeof(int)) {
            shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
        }
        // static_cast 在编译时就进行类型安全检查
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

    template <typename Dtype>
    Blob<Dtype>::Blob(const int num, const int channels, const int height, const int width)
        // capacity_ must be initialized before calling Reshape 
        // 因为capacity_在Reshape函数中会直接使用，因此需要提前初始化
        : capacity_(0) {
            Reshape(num, channels, height, width);
        }
    
    template <typename Dtype>
    Blobs<Dtype>::Blob(const vector<int>& shape)
        // capacity_ must be initialized before calling Reshape 
        // 因为capacity_在Reshape函数中会直接使用，因此需要提前初始化
        : capacity_(0) {
            Reshape(shape);
        }

    template <typename Dtype>
    const int* Blob<Dtype>::gpu_shape() const {
        CHECK(shape_data_);
        // 判断 shape_data_ 是否在GPU上，如果不在需要利用cuda将其搬运到GPU上
        return (const int*)shape_data_->gpu_data();
    }

    template <typename Dtype>
    const Dtype* Blob<Dtype>::cpu_data() const {
        CHECK(data_);
        // 判断 data_ 是否在CPU上，如果不在需要利用cuda将其搬运到CPU上
        return (const Dtype*) data_->cpu_data();
    }

    template <typename Dtype>
    void Blob<Dtype>::set_cpu_data(Dtype* data) {
        // 把指针data赋值给Blob.cpu_ptr_
        CHECK(data);
        // Make sure CPU and GPU sizes remains equal
        size_t size = count_ * sizeof(Dtype);
        if (data_->size()!=size) {
            data_.reset(new SyncedMemory(size));
            diff_.reset(new SyncedMemory(size));
        }
        data_->set_cpu_data(data);
    }

    template <typename Dtype>
    const Dtype* Blob<Dtype>::gpu_data() const {
        CHECK(data_);
        // 判断 shape_data_ 是否在GPU上，如果不在需要利用cuda将其搬运到GPU上
        return (const Dtype*) data_->gpu_data();
    }

    template <typename Dtype>
    void Blob<Dtype>::set_gpu_data(Dtype* data) {
        // 把指针data赋值给Blob.gpu_ptr_
        CHECK(data);
        // Make sure CPU and GPU sizes remains equal
        size_t size = count_ * sizeof(Dtype); // problem，为什么要比较这两个size
        if (data_.size()!=size) {
            data_.reset(new SyncedMemory(size));
            diff_.reset(new SyncedMemory(size));
        }
        data_->set_gpu_data(data);
    }

    template <typename Dtype>
    const Dtype* Blob<Dtype>::cpu_diff() const {
        CHECK(diff_);
        return (const Dtype*)diff_->cpu_data();
    }

    template <typename Dtype>
    const Dtype* Blob<Dtype>::gpu_diff() const {
        CHECK(diff_);
        return (const Dtype*)diff_->gpu_data();
    }

    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_cpu_data() {
        CHECK(data_);
        return static_cast<Dtype*> (data_->mutable_cpu_data());
    }

    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_gpu_data() {
        CHECK(data_);
        return static_cast<Dtype*>(data_->mutable_gpu_data());
    }

    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_cpu_diff() {
      CHECK(diff_);
      return static_cast<Dtype*>(diff_->mutable_cpu_data());
    }

    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_gpu_diff() {
      CHECK(diff_);
      return static_cast<Dtype*>(diff_->mutable_gpu_data());
    }

    template <typename Dtype>
    void Blob<Dtype>::ShareData(const Blob& other) {
        CHECK_EQ(count_, other.count());
        data_ = other.data();
    }

    template <typename Dtype>
    void Blob<Dtype>::ShareDiff(const Blob& other) {
        CHECK_EQ(count_, other.count_());
        diff_ = other.diff();
    }
    // The "update" method is used for parameter blobs in a Net, which are stored
    // as Blob<float> or Blob<double> -- hence we do not define it for
    // Blob<int> or Blob<unsigned int>.
    template <>
    void Blob<unsigned int>::Update() {
        NOT_IMPLEMENTED;
    }

    template <>
    void Blob<int>::Update() {
        NOT_IMPLEMENTED;
    }

    template <typename Dtype>
    void Blob<Dtype>::Update() {
        // y = y + (-1)*diff
        // 根据数据在CPU上还是GPU上分别调用不同的函数进行计算
        switch(data_->head()) {
            case SyncedMemory::HEAD_AT_CPU:
                // 在CPU上进行计算, y = y + (-1)*diff
                caffe_axpy<Dtype>(count_, Dtype(-1), static_cast<const Dtype*>(diff_->cpu_data()),
                                  static_cast<Dtype*>(data_->mutable_cpu_data()));
                break;
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
                #ifndef CPU_ONLY
                    caffe_gpu_axpy(count_, Dtype(-1), static_cast<const Dtype*>(diff_->gpu_data()),
                                   static_cast<Dtype*>(data_->mutable_gpu_data()));
                #else
                    NO_GPU;
                #endif

                break;
            default:
                LOG(FATAL) << "Syncedmem not initialized.";
        }
    }

    template <> unsigned int Blob<unsigned int>::asum_data() const {
      NOT_IMPLEMENTED;
      return 0;
    }

    template <> int Blob<int>::asum_data() const {
      NOT_IMPLEMENTED;
      return 0;
    }

    template<typename Dtype>
    Dtype Blob<Dtype>::asum_data() const {
        if(!data_) {return 0;}
        switch(data_->head()) {
            case SyncedMemory::HEAD_AT_CPU:
                return caffe_cpu_asum(count_, cpu_data());
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
                #ifndef CPU_ONLY
                    {
                        Dtype asum;
                        caffe_gpu_asum(count_, gpu_data(), &asum);
                        return asum;
                    }
                #else
                    NO_GPU;
                #endif
            case SyncedMemory::UNINITIALIZED:
                return 0;
            default:
                LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
        }
        return 0;
    }
    template <> unsigned int Blob<unsigned int>::asum_diff() const {
      NOT_IMPLEMENTED;
      return 0;
    }
    
    template <> int Blob<int>::asum_diff() const {
      NOT_IMPLEMENTED;
      return 0;
    }
    
    template <typename Dtype>
    Dtype Blob<Dtype>::asum_diff() const {
      if (!diff_) { return 0; }
      switch (diff_->head()) {
      case SyncedMemory::HEAD_AT_CPU:
        return caffe_cpu_asum(count_, cpu_diff());
      case SyncedMemory::HEAD_AT_GPU:
      case SyncedMemory::SYNCED:
    #ifndef CPU_ONLY
      {
        Dtype asum;
        caffe_gpu_asum(count_, gpu_diff(), &asum);
        return asum;
      }
    #else
        NO_GPU;
    #endif
      case SyncedMemory::UNINITIALIZED:
        return 0;
      default:
        LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
      }
      return 0;
    }
    
    template <> unsigned int Blob<unsigned int>::sumsq_data() const {
      NOT_IMPLEMENTED;
      return 0;
    }
    
    template <> int Blob<int>::sumsq_data() const {
      NOT_IMPLEMENTED;
      return 0;
    }
    
    template <typename Dtype>
    Dtype Blob<Dtype>::sumsq_data() const {
      Dtype sumsq;
      const Dtype* data;
      if (!data_) { return 0; }
      switch (data_->head()) {
      case SyncedMemory::HEAD_AT_CPU:
        data = cpu_data();
        sumsq = caffe_cpu_dot(count_, data, data);
        break;
      case SyncedMemory::HEAD_AT_GPU:
      case SyncedMemory::SYNCED:
    #ifndef CPU_ONLY
        data = gpu_data();
        caffe_gpu_dot(count_, data, data, &sumsq);
    #else
        NO_GPU;
    #endif
        break;
      case SyncedMemory::UNINITIALIZED:
        return 0;
      default:
        LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
      }
      return sumsq;
    }
    
    template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
      NOT_IMPLEMENTED;
      return 0;
    }
    
    template <> int Blob<int>::sumsq_diff() const {
      NOT_IMPLEMENTED;
      return 0;
    }
    
    template <typename Dtype>
    Dtype Blob<Dtype>::sumsq_diff() const {
      Dtype sumsq;
      const Dtype* diff;
      if (!diff_) { return 0; }
      switch (diff_->head()) {
      case SyncedMemory::HEAD_AT_CPU:
        diff = cpu_diff();
        sumsq = caffe_cpu_dot(count_, diff, diff);
        break;
      case SyncedMemory::HEAD_AT_GPU:
      case SyncedMemory::SYNCED:
    #ifndef CPU_ONLY
        diff = gpu_diff();
        caffe_gpu_dot(count_, diff, diff, &sumsq);
        break;
    #else
        NO_GPU;
    #endif
      case SyncedMemory::UNINITIALIZED:
        return 0;
      default:
        LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
      }
      return sumsq;
    }
    
    template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
      NOT_IMPLEMENTED;
    }
    
    template <> void Blob<int>::scale_data(int scale_factor) {
      NOT_IMPLEMENTED;
    }
    
    template <typename Dtype>
    void Blob<Dtype>::scale_data(Dtype scale_factor) {
      Dtype* data;
      if (!data_) { return; }
      switch (data_->head()) {
      case SyncedMemory::HEAD_AT_CPU:
        data = mutable_cpu_data();
        caffe_scal(count_, scale_factor, data);
        return;
      case SyncedMemory::HEAD_AT_GPU:
      case SyncedMemory::SYNCED:
    #ifndef CPU_ONLY
        data = mutable_gpu_data();
        caffe_gpu_scal(count_, scale_factor, data);
        return;
    #else
        NO_GPU;
    #endif
      case SyncedMemory::UNINITIALIZED:
        return;
      default:
        LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
      }
    }
    
    template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
      NOT_IMPLEMENTED;
    }
    
    template <> void Blob<int>::scale_diff(int scale_factor) {
      NOT_IMPLEMENTED;
    }
    
    template <typename Dtype>
    void Blob<Dtype>::scale_diff(Dtype scale_factor) {
      Dtype* diff;
      if (!diff_) { return; }
      switch (diff_->head()) {
      case SyncedMemory::HEAD_AT_CPU:
        diff = mutable_cpu_diff();
        caffe_scal(count_, scale_factor, diff);
        return;
      case SyncedMemory::HEAD_AT_GPU:
      case SyncedMemory::SYNCED:
    #ifndef CPU_ONLY
        diff = mutable_gpu_diff();
        caffe_gpu_scal(count_, scale_factor, diff);
        return;
    #else
        NO_GPU;
    #endif
      case SyncedMemory::UNINITIALIZED:
        return;
      default:
        LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
      }
    }

    template <typename Dtype>
    bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
      // BlobProto 定义在 caffe.pb.h，由 ./src/proto/caffe.proto 通过 protoc 解码生成
      // 可以通过看 caffe.proto 里面的key_val 内容看
      if (other.has_num() || other.has_channels() ||
          other.has_height() || other.has_width()) {
        // Using deprecated 4D Blob dimensions --
        // shape is (num, channels, height, width).
        // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
        // methods as these index from the beginning of the blob shape, where legacy
        // parameter blobs were indexed from the end of the blob shape (e.g., bias
        // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
        return shape_.size() <= 4 &&
               LegacyShape(-4) == other.num() &&
               LegacyShape(-3) == other.channels() &&
               LegacyShape(-2) == other.height() &&
               LegacyShape(-1) == other.width();
      }
      vector<int> other_shape(other.shape().dim_size());
      for (int i = 0; i < other.shape().dim_size(); ++i) {
        other_shape[i] = other.shape().dim(i);
      }
      return shape_ == other_shape;
    }

    template <typename Dtype>
    void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
      if(source.count() != count_ || source.shape() != shape_) {
        if (reshape) {
          ReshapeLike(source);
        }
        else {
          LOG(FATAL) << "Trying to copy blobs of different sizes.";
        }
      }

      switch(Caffe::mode()) {
        case Caffe::GPU:
          if (copy_diff) {
            caffe_copy(count_, source.gpu_diff(), static_cast<Dtype*>(diff_->mutable_gpu_data()));
          }
          else {
            caffe_copy(count_, source.gpu_data(), static_cast<Dytpe*>(data_->mutable_gpu_data()));
          }
          break;
        case Caffe::CPU:
          if (copy_diff) {
            caffe_copy(count_, source.cpu_diff(), static_cast<Dtype*>(diff_->mutable_cpu_data()));
          }
          else {
            caffe_copy(count_, source.cpu_diff(), static_cast<Dtype*>(data_->mutable_cpu_data()));
          }
          break;

        default:
          LOG(FATAL) << " Unkown caffe mode.";
      }
    }

