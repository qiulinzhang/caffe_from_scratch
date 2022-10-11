#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include<algorithm>
#include<string>
#include<vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */

 namespace boost {class mutex;}
 namespace caffe {
 /**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with
 * their output Blob%s.
 */
 template <typename Dtype>
 class Layer {
    public:
    /**
    * You should not implement your own constructor. Any set up code should go
    * to SetUp(), where the dimensions of the bottom blobs are provided to the
    * layer.
    */
        explicit Layer(const LayerParameter& param) : layer_param_(param) {
            // Set phase and copy blobs (if there are any)
            phase_ = param.phase();
            if(layer_param_.blob_size() > 0) {
                blobs_.resize(layer_param_.blob_size());
                for(int i=0; i<layer_param_.blobs_.blob_size(); ++i){
                    blobs_[i].reset(new Blob<Dtype>()); // blobs_[i] 的类型是 shared_ptr<Blob<Dtype>>
                    blobs_[i]->FromProto(layer_param_.blobs_[i]);
                }
            }
        }

        virtual ~Layer() {}

        /**
        * @brief Implements common layer setup functionality.
        *
        * @param bottom the preshaped input blobs
        * @param top
        *     the allocated but unshaped output blobs, to be shaped by Reshape
        *
        * Checks that the number of bottom and top blobs is correct.
        * Calls LayerSetUp to do special layer setup for individual layer types,
        * followed by Reshape to set up sizes of top blobs and internal buffers.
        * Sets up the loss weight multiplier blobs for any non-zero loss weights.
        * This method may not be overridden.
        */
        void SetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype*>& top) {
            CheckBlobCounts(bottom, top);
            LayerSetUp(bottom, top);
            Reshape(bottom, top);
            SetLossWeights(top);
        }

        /**
        * @brief Does layer-specific setup: your layer should implement this function
        *        as well as Reshape.
        *
        * @param bottom
        *     the preshaped input blobs, whose data fields store the input data for
        *     this layer
        * @param top
        *     the allocated but unshaped output blobs
        *
        * This method should do one-time layer specific setup. This includes reading
        * and processing relevent parameters from the <code>layer_param_</code>.
        * Setting up the shapes of top blobs and internal buffers should be done in
        * <code>Reshape</code>, which will be called before the forward pass to
        * adjust the top blob sizes.
        */
        virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> & top) {}

        /**
        * @brief Adjust the shapes of top blobs and internal buffers to accommodate
        *        the shapes of the bottom blobs.
        *
        * @param bottom the input blobs, with the requested input shapes
        * @param top the top blobs, which should be reshaped as needed
        *
        * This method should reshape top blobs as needed according to the shapes
        * of the bottom (input) blobs, as well as reshaping any internal buffers
        * and making any other necessary adjustments so that the layer can
        * accommodate the bottom blobs.
        */
        virtual void Reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*>&top) = 0;

        /**
        * @brief Given the bottom blobs, compute the top blobs and the loss.
        *
        * @param bottom
        *     the input blobs, whose data fields store the input data for this layer
        * @param top
        *     the preshaped output blobs, whose data fields will store this layers'
        *     outputs
        * \return The total loss from the layer.
        *
        * The Forward wrapper calls the relevant device wrapper function
        * (Forward_cpu or Forward_gpu) to compute the top blob values given the
        * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
        * then computes and returns the loss.
        *
        * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
        */
        inline Dtype Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);


        /**
         * @brief Given the top blob error gradients, compute the bottom blob error
         *        gradients.
         *
         * @param top
         *     the output blobs, whose diff fields store the gradient of the error
         *     with respect to themselves
         * @param propagate_down
         *     a vector with equal length to bottom, with each index indicating
         *     whether to propagate the error gradients down to the bottom blob at
         *     the corresponding index
         * @param bottom
         *     the input blobs, whose diff fields will store the gradient of the error
         *     with respect to themselves after Backward is run
         *
         * The Backward wrapper calls the relevant device wrapper function
         * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
         * top blob diffs.
         *
         * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
         */
        inline void Backward(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);

        /**
        * @brief Returns the vector of learnable parameter blobs.
        */

        vector<shaped_ptr<Blob<Dtype> >& blobs() {
            return blobs_;
        }

    protected:
        /** The protobuf that stores the layer parameters */
        LayerParameter layer_param_;
        /** The phase: TRAIN or TEST */
        Phase phase_;
        /** The vector that stores the learnable parameters as a set of blobs. */
        vector<shared_ptr<Blob<Dtype> > > blobs_;
        /** Vector indicating whether to compute the diff of each param blob. */
        vector<bool> param_propagate_down_;
        /** The vector that indicates whether each top blob has a non-zero weight in
         *  the objective function. */
        vector<Dtype> loss_;

        /** @brief Using the CPU device, compute the layer output. */
        virtual void Forward_cpu(const vector<Blob<Dtype>* >&bottom, const vector<Blob<Dtype>*> & top) = 0;
        virtual void Forward_gpu(const vector<Blob<Dtype>* >&bottom, const vector<Blob<Dtype>*> & top) {
            // LOG(WARNING) <<"Using CPU code as backup.";
            return Forward_cpu(bottom, top);
        }

        /**
          * @brief Using the CPU device, compute the gradients for any parameters and
          *        for the bottom blobs if propagate_down is true.
          */
        virtual void Backward_cpu(const vector<Blob<Dtype>*> &top,
                                  const vector<bool>& propagate_down,
                                  const vector<Blob<Dtype>*>& bottom) = 0;

        /**
        * @brief Using the GPU device, compute the gradients for any parameters and
        *        for the bottom blobs if propagate_down is true.
        *        Fall back to Backward_cpu() if unavailable.
        */
        virtual void Backward_gpu(const vector<Blob<Dtype>*>&top,
                                  const vector<bool>& propagate_down,
                                  const vector<Blob<Dtype>*>& bottom) {
                                  // LOG(WARNING) << "Using CPU code as backup.";
                                  Backward_gpu(top, propagate_down, bottom);
                                  }
 }
 }