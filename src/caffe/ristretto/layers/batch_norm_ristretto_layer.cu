#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
void BatchNormRistrettoLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Trim layer input
  if (this->phase_ == TEST) {
      this->QuantizeLayerInputs_gpu(bottom[0]->mutable_gpu_data(),
          bottom[0]->count());
  }

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(this->channels_*bottom[0]->shape(0));

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
  // Trim rounding_ setting********************************
  int rounding = this->phase_ == TEST ? this->rounding_ :
      QuantizationParameter_Rounding_STOCHASTIC;

  if (this->use_global_stats_) {
    //Dtype* blob_data = this->blobs_[2]->cpu_data();
    //for(int num_i = 0;num_i <this->blobs_[2]->count();num_i++ ){
      // LOG(INFO) << "debugOutput "<< "layer name : "<< this->layer_param_.name();
      // LOG(INFO) << "debugOutput "<< "this->bw_mean_ : "<< this->bw_mean_;
      // LOG(INFO) << "debugOutput "<< "this->fl_mean_ : "<< this->fl_mean_;
      // LOG(INFO) << "debugOutput "<< "this->bw_var_ : "<< this->bw_var_;
      // LOG(INFO) << "debugOutput "<< "this->fl_var_ : "<< this->fl_var_;
      // LOG(INFO) << "debugOutput "<< "this->bw_mov : "<< this->bw_mov_;
      // LOG(INFO) << "debugOutput "<< "this->fl_mov_ : "<< this->fl_mov_;
      // LOG(INFO) << "debugOutput " << "blobs_[2]= " <<this->blobs_[2]->cpu_data()[0];
    //}
    // Trim scale_factor********************************//
    caffe_copy(this->blobs_[2]->count(), this->blobs_[2]->cpu_data(),
        this->weights_quantized_[2]->mutable_cpu_data());
    this->QuantizeWeights_gpu(this->weights_quantized_[2]->mutable_cpu_data(),this->weights_quantized_[2]->count(), rounding,this->type(),2);
    //for(int num_i = 0;num_i <this->weights_quantized_[2]->count();num_i++ ){
    //  LOG(INFO) << "debugOutput " << "weights_quantized_[2]= " << this->weights_quantized_[2]->cpu_data()[0];
    //}
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->weights_quantized_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->weights_quantized_[2]->cpu_data()[0];
    caffe_gpu_scale(this->variance_.count(), scale_factor,
        this->blobs_[0]->gpu_data(), this->mean_.mutable_gpu_data());
    caffe_gpu_scale(this->variance_.count(), scale_factor,
        this->blobs_[1]->gpu_data(), this->variance_.mutable_gpu_data());
  } else {
    // compute mean
    caffe_gpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim,
        1. / (num * spatial_dim), bottom_data,
        this->spatial_sum_multiplier_.gpu_data(), 0.,
        this->num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1.,
        this->num_by_chans_.gpu_data(), this->batch_sum_multiplier_.gpu_data(), 0.,
        this->mean_.mutable_gpu_data());
  }

  // Trim mean********************************//
  caffe_copy(this->mean_.count(), this->mean_.gpu_data(),
      this->weights_quantized_[0]->mutable_gpu_data());
  this->QuantizeWeights_gpu(this->weights_quantized_[0]->mutable_gpu_data(),this->weights_quantized_[0]->count(), rounding,this->type(),0);

  // subtract mean
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1,
      this->batch_sum_multiplier_.gpu_data(), this->weights_quantized_[0]->gpu_data(), 0.,
      this->num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num,
      spatial_dim, 1, -1, this->num_by_chans_.gpu_data(),
      this->spatial_sum_multiplier_.gpu_data(), 1., top_data);

  if (!this->use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_gpu_powx(top[0]->count(), top_data, Dtype(2),
        this->temp_.mutable_gpu_data());  // (X-EX)^2
    caffe_gpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim,
        1. / (num * spatial_dim), this->temp_.gpu_data(),
        this->spatial_sum_multiplier_.gpu_data(), 0.,
        this->num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1.,
        this->num_by_chans_.gpu_data(), this->batch_sum_multiplier_.gpu_data(), 0.,
        this->variance_.mutable_gpu_data());  // E((X_EX)^2)

    // compute and save moving average
    this->blobs_[2]->mutable_cpu_data()[0] *= this->moving_average_fraction_;
    this->blobs_[2]->mutable_cpu_data()[0] += 1;
    caffe_gpu_axpby(this->mean_.count(), Dtype(1), this->mean_.gpu_data(),
        this->moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
    int m = bottom[0]->count()/this->channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    caffe_gpu_axpby(this->variance_.count(), bias_correction_factor,
        this->variance_.gpu_data(), this->moving_average_fraction_,
        this->blobs_[1]->mutable_gpu_data());
  }

  // Trim variance_********************************//
  caffe_copy(this->variance_.count(), this->variance_.gpu_data(),
        this->weights_quantized_[1]->mutable_gpu_data());
  this->QuantizeWeights_gpu(this->weights_quantized_[1]->mutable_gpu_data(),this->weights_quantized_[1]->count(), rounding,this->type(),1);

  // normalize variance
  caffe_gpu_add_scalar(this->weights_quantized_[1]->count(), this->eps_, this->weights_quantized_[1]->mutable_gpu_data());
  caffe_gpu_powx(this->weights_quantized_[1]->count(), this->weights_quantized_[1]->gpu_data(), Dtype(0.5),
      this->weights_quantized_[1]->mutable_gpu_data());

  // replicate variance to input size
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1,
      this->batch_sum_multiplier_.gpu_data(), this->weights_quantized_[1]->gpu_data(), 0.,
      this->num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num,
      spatial_dim, 1, 1., this->num_by_chans_.gpu_data(),
      this->spatial_sum_multiplier_.gpu_data(), 0., this->temp_.mutable_gpu_data());
  caffe_gpu_div(this->temp_.count(), top_data, this->temp_.gpu_data(), top_data);

  // Trim layer output***************//
  if (this->phase_ == TEST) {
    this->QuantizeLayerOutputs_gpu(top_data, top[0]->count());
  }

  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(this->x_norm_.count(), top_data,
      this->x_norm_.mutable_gpu_data());
}

template <typename Dtype>
void BatchNormRistrettoLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->gpu_diff();
  } else {
    caffe_copy(this->x_norm_.count(), top[0]->gpu_diff(), this->x_norm_.mutable_gpu_diff());
    top_diff = this->x_norm_.gpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  if (this->use_global_stats_) {
    caffe_gpu_div(this->temp_.count(), top_diff, this->temp_.gpu_data(), bottom_diff);
    return;
  }
  const Dtype* top_data = this->x_norm_.gpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count()/(this->channels_*bottom[0]->shape(0));
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_gpu_mul(this->temp_.count(), top_data, top_diff, bottom_diff);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim, 1.,
      bottom_diff, this->spatial_sum_multiplier_.gpu_data(), 0.,
      this->num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1.,
      this->num_by_chans_.gpu_data(), this->batch_sum_multiplier_.gpu_data(), 0.,
      this->weights_quantized_[0]->mutable_gpu_data());

  // reshape (broadcast) the above
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1,
      this->batch_sum_multiplier_.gpu_data(), this->weights_quantized_[0]->gpu_data(), 0.,
      this->num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num,
      spatial_dim, 1, 1., this->num_by_chans_.gpu_data(),
      this->spatial_sum_multiplier_.gpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_mul(this->temp_.count(), top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim, 1.,
      top_diff, this->spatial_sum_multiplier_.gpu_data(), 0.,
      this->num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1.,
      this->num_by_chans_.gpu_data(), this->batch_sum_multiplier_.gpu_data(), 0.,
      this->weights_quantized_[0]->mutable_gpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1,
      this->batch_sum_multiplier_.gpu_data(), this->weights_quantized_[0]->gpu_data(), 0.,
      this->num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * this->channels_,
      spatial_dim, 1, 1., this->num_by_chans_.gpu_data(),
      this->spatial_sum_multiplier_.gpu_data(), 1., bottom_diff);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_gpu_axpby(this->temp_.count(), Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff);

  // note: this->temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_gpu_div(this->temp_.count(), bottom_diff, this->temp_.gpu_data(), bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormRistrettoLayer);


}  // namespace caffe
