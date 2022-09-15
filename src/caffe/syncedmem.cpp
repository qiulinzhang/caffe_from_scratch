#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
// 主要是GPU上的内存分配，以及CPU和GPU之间的数据拷贝
namespace caffe{
    SyncedMemory::SyncedMemory(): cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0),
                                  head_(UNINITIALIZED), own_cpu_data_(false),
                                  cpu_malloc_use_cuda_(false),
                                  own_gpu_data_(false){
                    #ifndef CPU_ONLY
                        #ifdef DEBUG
                            CUDA_CHECK(cudaGetDevice(&device));
                        #endif
                    #endif
                    }
    SyncedMemory::~SyncedMemory(){
        check_device();
        if (cpu_ptr_ && own_cpu_data_){
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
        #ifndef CPU_ONLY
            if (gpu_ptr_ && own_gpu_data_){
                CUDA_CHECK(cudaFree(gpu_ptr_));
            }
        #endif // CPU_ONLY
    }


    inline void SyncedMemory::to_cpu() {
        check_device();
        switch(head_){
            case UNINITIALIZED:
                CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
                caffe_memset(size_, 0, cpu_ptr_);
                head_ = HEAD_AT_CPU;
                own_cpu_data_ = true;
                break;
            case HEAD_AT_GPU:
                #ifndef CPU_ONLY
                    if (cpu_ptr_ == NULL){
                        CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
                        own_cpu_data_ = true;
                    }
                    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
                    head_ = SYNCED;
                #else
                    NO_GPU;
                #endif
                    break;
            case HEAD_AT_CPU:
            case SYNCED:
                break;
        }
    }

    inline void SyncedMemory::to_gpu() {
        check_device();
        #ifndef CPU_ONLY
            switch (head_) {
                case UNINITIALIZED:
                    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
                    caffe_gpu_memset(size_, 0, gpu_ptr_);
                    head_ = HEAD_AT_GPU;
                    own_gpu_data_ = true;
                    break;
                case HEAD_AT_CPU:
                    if (gpu_ptr_ == NULL){
                        CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
                        own_gpu_data_ = true;
                    }
                    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
                    head_ = SYNCED;
                    break;
                case HEAD_AT_GPU:
                case SYNCED:
                    break;
            }
        #else
            NO_GPU;
        #endif
    }
    // problem，与 mutable_cpu_data的区别，(const void*)的作用是啥？
    // void* 是指返回一个指针，但是指针的类型不受限制，可以直接返回 int or float 型的指针 
    // const 表示其值不发生变化
    const void* SyncedMemory::cpu_data(){
        // 返回值被const修饰，表示返回值不可被改变
        check_device();
        to_cpu();
        return (const void*) cpu_ptr_;
    }

    void SyncedMemory::set_cpu_data(void* data){
        check_device();
        CHECK(data);
        if (own_cpu_data_){
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
        cpu_ptr_ = data;
        head_ = HEAD_AT_CPU;
        own_cpu_data_ = false; // problem 为什么有了cpu_data还要将其设置为False
    }

    const void* SyncedMemory::gpu_data() {
        check_device();
        #ifndef CPU_ONLY
            to_gpu();
            reurn (const void*) gpu_ptr_;
        #else
            NO_GPU;
            return NULL;
        #endif
    }

    void SyncedMemory::set_gpu_data(void* data){
        check_device();
        #ifndef CPU_ONLY
            CHECK(data);
            if (own_gpu_data_){
                CUDA_CHECK(cudaFree(gpu_ptr_));
            }
            gpu_ptr_ = data;
            head_ = HEAD_AT_GPU;
            own_gpu_data_ = false; //problem
        #else
            NO_GPU;
        #endif
    }

    void* SyncedMemory::mutable_cpu_data(){
        // 返回值没有被const修饰，表示返回值可被改变，对应于 mutable(易变化的，多变的)
        check_device();
        to_cpu();
        head_ = HEAD_AT_CPU;
        return cpu_ptr_;
    }

    void* SyncedMemory::mutable_gpu_data() {
        check_device();
        #ifndef CPU_ONLY
            to_gpu();
            head_ = HEAD_AT_GPU;
            return gpu_ptr_;
        #else
            NO_GPU;
            return NULL;
        #endif
    }

    #ifndef CPU_ONLY
        void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
            check_device();
            CHECK(head_ == HEAD_AT_CPU);
            if(gpu_ptr_ == NULL){
                CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
                own_gpu_data_ = true;
            }
            const cudaMemcpyKind put = cudaMemcpyHostToDevice;
            CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
            // Assume caller will synchronize on the stream before use
            head_ = SYNCED;
        }
    #endif

    void SyncedMemory::check_device() {
        #ifndef CPU_ONLY
            #ifdef DEBUG
                int device;
                cudaGetDeivce(&device);
                CHECK(device == device_);// problem 没有初始化的device直接进行赋值判断？？
                if (gpu_ptr_ && own_gpu_data_){
                    cudaPointerAttributes attributes;
                    CDUA_CHECK(cudaPointerAttributes(&attributes, gpu_ptr_));
                    CHECK(attributes.device == device_);
                }
            #endif
        #endif
    }
} // namespace caffe