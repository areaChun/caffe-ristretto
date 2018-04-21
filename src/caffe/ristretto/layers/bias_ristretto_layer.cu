#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BiasForward(const int n, const Dtype* in,
    const Dtype* bias, const int bias_dim, const int inner_dim,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int bias_index = (index / inner_dim) % bias_dim;
    out[index] = in[index] + bias[bias_index];
  }
}

template <typename Dtype>
void BiasRistrettoLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Trim layer input
  // if (this->phase_ == TEST) {
  //     this->QuantizeLayerInputs_gpu(bottom[0]->mutable_gpu_data(),
  //         bottom[0]->count());
  // }
  // Trim weights
  LOG(INFO) << "debugOutput "<<"BiasRistrettoLayer<Dtype>::Forward_gpu";
  caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
      this->weights_quantized_[0]->mutable_gpu_data());
  int rounding = this->phase_ == TEST ? this->rounding_ :
      QuantizationParameter_Rounding_STOCHASTIC;
  this->QuantizeWeights_gpu(this->weights_quantized_[0]->mutable_gpu_data(),this->weights_quantized_[0]->count(), 
      rounding,this->type(),0);

  const int count = top[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bias_data =
      ((bottom.size() > 1) ? bottom[1] : this->weights_quantized_[0].get())->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  BiasForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bias_data, this->bias_dim_, this->inner_dim_, top_data);

    // Trim layer output
  // if (this->phase_ == TEST) {
  //   this->QuantizeLayerOutputs_gpu(top_data, top[0]->count());
  // }
}

template <typename Dtype>
void BiasRistrettoLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] && bottom[0] != top[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }
  // in-place, we don't need to do anything with the data diff
  const bool bias_param = (bottom.size() == 1);
  if ((!bias_param && propagate_down[1]) ||
      (bias_param && this->param_propagate_down_[0])) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bias_diff = (bias_param ? this->blobs_[0].get() : bottom[1])
        ->mutable_gpu_diff();
    bool accum = bias_param;
    for (int n = 0; n < this->outer_dim_; ++n) {
      caffe_gpu_gemv(CblasNoTrans, this->bias_dim_, this->inner_dim_, Dtype(1),
          top_diff, this->bias_multiplier_.gpu_data(), Dtype(accum), bias_diff);
      top_diff += this->dim_;
      accum = true;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BiasRistrettoLayer);

}  // namespace caffe
