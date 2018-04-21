#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
BiasRistrettoLayer<Dtype>::BiasRistrettoLayer(const LayerParameter& param)
      : BiasLayer<Dtype>(param), BaseRistrettoLayer<Dtype>() {
  LOG(INFO) << "debugOutput "<<"BiasRistrettoLayer()";
  this->precision_ = this->layer_param_.quantization_param().precision();
  this->rounding_ = this->layer_param_.quantization_param().rounding_scheme();
  switch (this->precision_) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
    this->bw_params_ = this->layer_param_.quantization_param().bw_params();
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
    this->fl_params_ = this->layer_param_.quantization_param().fl_params();
    break;
  case QuantizationParameter_Precision_MINIFLOAT:
    this->fp_mant_ = this->layer_param_.quantization_param().mant_bits();
    this->fp_exp_ = this->layer_param_.quantization_param().exp_bits();
    break;
  case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
    this->pow_2_min_exp_ = this->layer_param_.quantization_param().exp_min();
    this->pow_2_max_exp_ = this->layer_param_.quantization_param().exp_max();
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
    break;
  default:
    LOG(FATAL) << "Unknown precision mode: " << this->precision_;
    break;
  }
}

template <typename Dtype>
void BiasRistrettoLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // bias is a learned parameter; initialize it
    const BiasParameter& param = this->layer_param_.bias_param();
    const int axis = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), axis + num_axes)
          << "bias blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis;
    }
    this->blobs_.resize(1);
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + axis;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> bias_shape(shape_start, shape_end);
    this->blobs_[0].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(param.filler()));
    filler->Fill(this->blobs_[0].get());
  }// parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

    // Prepare quantized weights
  this->weights_quantized_.resize(1);
  const BiasParameter& param = this->layer_param_.bias_param();
  const int axis = bottom[0]->CanonicalAxisIndex(param.axis());
  const int num_axes = param.num_axes();
  CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                         << "or -1 to extend to the end of bottom[0]";
  if (num_axes >= 0) {
    CHECK_GE(bottom[0]->num_axes(), axis + num_axes)
        << "bias blob's shape extends past bottom[0]'s shape when applied "
        << "starting with bottom[0] axis = " << axis;
  }
  LOG(INFO) << "debugOutput "<<"this->LayerSetUp";
  const vector<int>::const_iterator& shape_start =
      bottom[0]->shape().begin() + axis;
  const vector<int>::const_iterator& shape_end =
      (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
  vector<int> bias_shape(shape_start, shape_end);
  this->weights_quantized_[0].reset(new Blob<Dtype>(bias_shape));
}

// template <typename Dtype>
// void BiasRistrettoLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top) {
//   const BiasParameter& param = this->layer_param_.bias_param();
//   Blob<Dtype>* bias = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
//   // Always set axis == 0 in special case where bias is a scalar
//   // (num_axes == 0). Mathematically equivalent for any choice of axis, so the
//   // actual setting can be safely ignored; and computation is most efficient
//   // with axis == 0 and (therefore) outer_dim_ == 1.
//   const int axis = (bias->num_axes() == 0) ?
//       0 : bottom[0]->CanonicalAxisIndex(param.axis());
//   CHECK_GE(bottom[0]->num_axes(), axis + bias->num_axes())
//       << "bias blob's shape extends past bottom[0]'s shape when applied "
//       << "starting with bottom[0] axis = " << axis;
//   for (int i = 0; i < bias->num_axes(); ++i) {
//     CHECK_EQ(bottom[0]->shape(axis + i), bias->shape(i))
//         << "dimension mismatch between bottom[0]->shape(" << axis + i
//         << ") and bias->shape(" << i << ")";
//   }
//   outer_dim_ = bottom[0]->count(0, axis);
//   bias_dim_ = bias->count();
//   inner_dim_ = bottom[0]->count(axis + bias->num_axes());
//   dim_ = bias_dim_ * inner_dim_;
//   if (bottom[0] != top[0]) {
//     top[0]->ReshapeLike(*bottom[0]);
//   }
//   bias_multiplier_.Reshape(vector<int>(1, inner_dim_));
//   if (bias_multiplier_.cpu_data()[inner_dim_ - 1] != Dtype(1)) {
//     caffe_set(inner_dim_, Dtype(1), bias_multiplier_.mutable_cpu_data());
//   }
// }

template <typename Dtype>
void BiasRistrettoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bias_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
  LOG(INFO) << "debugOutput "<<"Dtype* top_data";
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (bottom[0] != top[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
  LOG(INFO) << "for (int n = 0; n < outer_dim_; ++n)";
  for (int n = 0; n < outer_dim_; ++n) {
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bias_dim_,
        inner_dim_, 1, Dtype(1), bias_data,
        bias_multiplier_.cpu_data(), Dtype(1), top_data);
    top_data += dim_;
  }
  LOG(INFO) << "top_data += dim_;";
}

// template <typename Dtype>
// void BiasRistrettoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top) {
//   //*      follow the scale layer,so only quantize the param not input & output     *//
//   //   // Trim layer input
//   // if (this->phase_ == TEST) {
//   //     this->QuantizeLayerInputs_cpu(bottom[0]->mutable_cpu_data(),
//   //         bottom[0]->count());
//   // }
//   // Trim weights
//   LOG(INFO) << "debugOutput "<<"BiasRistrettoLayer<Dtype>::Forward_cpu";
//   caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
//       this->weights_quantized_[0]->mutable_cpu_data());
//   int rounding = this->phase_ == TEST ? this->rounding_ :
//       QuantizationParameter_Rounding_STOCHASTIC;
//   LOG(INFO) << "debugOutput "<<"his->QuantizeWeights_cpu";
//   this->QuantizeWeights_cpu(this->weights_quantized_[0]->mutable_cpu_data(),this->weights_quantized_[0]->count(), 
//       rounding,this->type(),0);
//   LOG(INFO) << "debugOutput "<<"bias_data";
//   const Dtype* bias_data =
//       ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
//   Dtype* top_data = top[0]->mutable_cpu_data();
//   LOG(INFO) << "debugOutput "<<"if (bottom[0] != top[0])";
//   if (bottom[0] != top[0]) {
//     const Dtype* bottom_data = bottom[0]->cpu_data();
//     caffe_copy(bottom[0]->count(), bottom_data, top_data);
//   }
//   LOG(INFO) << "debugOutput "<<"caffe_cpu_gemm";
//   for (int n = 0; n < this->outer_dim_; ++n) {
//     caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, this->bias_dim_,
//         this->inner_dim_, 1, Dtype(1), bias_data,
//         this->bias_multiplier_.cpu_data(), Dtype(1), top_data);
//     top_data += this->dim_;
//   }
//   LOG(INFO) << "debugOutput "<<"caffe_cpu_gemm end";

//   // // Trim layer output
//   // if (this->phase_ == TEST) {
//   //   this->QuantizeLayerOutputs_cpu(top_data, top[0]->count());
//   // }
// }

template <typename Dtype>
void BiasRistrettoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] && bottom[0] != top[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }
  // in-place, we don't need to do anything with the data diff
  const bool bias_param = (bottom.size() == 1);
  if ((!bias_param && propagate_down[1]) ||
      (bias_param && this->param_propagate_down_[0])) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bias_diff = (bias_param ? this->blobs_[0].get() : bottom[1])
        ->mutable_cpu_diff();
    bool accum = bias_param;
    for (int n = 0; n < outer_dim_; ++n) {
      caffe_cpu_gemv(CblasNoTrans, bias_dim_, inner_dim_, Dtype(1),
          top_diff, bias_multiplier_.cpu_data(), Dtype(accum), bias_diff);
      top_diff += dim_;
      accum = true;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BiasRistrettoLayer);
#endif

INSTANTIATE_CLASS(BiasRistrettoLayer);
REGISTER_LAYER_CLASS(BiasRistretto);

}  // namespace caffe
