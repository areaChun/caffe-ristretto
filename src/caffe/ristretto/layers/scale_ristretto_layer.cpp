#include <algorithm>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
ScaleRistrettoLayer<Dtype>::ScaleRistrettoLayer(const LayerParameter& param)
      : ScaleLayer<Dtype>(param), BaseRistrettoLayer<Dtype>() {
  LOG(INFO) << "debugOutput "<<"ScaleRistrettoLayer()";
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
void ScaleRistrettoLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // scale is a learned parameter; initialize it
    this->axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), this->axis_ + num_axes)
          << "scale blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << this->axis_;
    }
    this->blobs_.resize(1);
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + this->axis_;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> scale_shape(shape_start, shape_end);
    this->blobs_[0].reset(new Blob<Dtype>(scale_shape));

    //install weights_quantized
    this->weights_quantized_.resize(1);
    this->weights_quantized_[0].reset(new Blob<Dtype>(scale_shape));

    FillerParameter filler_param(param.filler());
    if (!param.has_filler()) {
      // Default to unit (1) filler for identity operation.
      filler_param.set_type("constant");
      filler_param.set_value(1);
    }
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[0].get());
  }
  if (param.bias_term()) {
    LayerParameter layer_param(this->layer_param_);
    layer_param.set_type("Bias");
    // layer_param.set_type("BiasRistretto");
    // LOG(INFO) << "debugOutput "<<"set_type(BiasRistrettoLayer)";
    BiasParameter* bias_param = layer_param.mutable_bias_param();
    // QuantizationParameter* bias_quantization_param = layer_param.mutable_quantization_param();
    // switch (this->precision_) {
    // case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    //   bias_quantization_param->set_bw_params(this->bw_params_);
    //   bias_quantization_param->set_fl_params(this->fl_params_);
    //   break;
    // case QuantizationParameter_Precision_MINIFLOAT:
    //   bias_quantization_param->set_mant_bits(this->fp_mant_);
    //   bias_quantization_param->set_exp_bits(this->fp_exp_);
    //   break;
    // case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
    //   bias_quantization_param->set_exp_min(this->pow_2_min_exp_);
    //   bias_quantization_param->set_exp_max(this->pow_2_max_exp_);
    //   break;
    // default:
    //   LOG(FATAL) << "Unknown precision mode: " << this->precision_;
    //   break;
    // }
    bias_param->set_axis(param.axis());
    if (bottom.size() > 1) {
      bias_param->set_num_axes(bottom[1]->num_axes());
    } else {
      bias_param->set_num_axes(param.num_axes());
    }
    bias_param->mutable_filler()->CopyFrom(param.bias_filler());
    this->bias_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    this->bias_bottom_vec_.resize(1);
    this->bias_bottom_vec_[0] = bottom[0];
    this->bias_layer_->SetUp(this->bias_bottom_vec_, top);
    this->bias_param_id_ = this->blobs_.size();
    this->blobs_.resize(this->bias_param_id_ + 1);
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), this->axis_ + num_axes)
          << "scale blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << this->axis_;
    }
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + this->axis_;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> scale_shape(shape_start, shape_end);
    this->blobs_[this->bias_param_id_].reset(new Blob<Dtype>(scale_shape));
    //this->blobs_[this->bias_param_id_] = this->bias_layer_->blobs()[0];
    this->bias_propagate_down_.resize(1, false);

    //install weights_quantized
    this->weights_quantized_.resize(this->bias_param_id_ + 1);
    this->weights_quantized_[this->bias_param_id_] = this->bias_layer_->blobs()[0];
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}


template <typename Dtype>
void ScaleRistrettoLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // // Trim layer input
  //   LOG(INFO) << "debugOutput "<<"if (this->phase_ == TEST)";
  // if (this->phase_ == TEST) {
  //     this->QuantizeLayerInputs_cpu(bottom[0]->mutable_cpu_data(),
  //         bottom[0]->count());
  // }
  // // Trim weights
  // caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
  //     this->weights_quantized_[0]->mutable_cpu_data());

  // LOG(INFO) << "debugOutput "<<"this->QuantizeWeights_cpu";
  // //*******************************************************************************quantized bias_layer
  // int rounding = this->phase_ == TEST ? this->rounding_ :
  //     QuantizationParameter_Rounding_STOCHASTIC;
  // this->QuantizeWeights_cpu(this->weights_quantized_[0]->mutable_cpu_data(),
  //   this->weights_quantized_[0]->count(), rounding,this->type(),0);
  // // this->QuantizeWeights_cpu(this->weights_quantized_, rounding,
  // //     this->bias_term_);
  // LOG(INFO) << "debugOutput "<<"  const Dtype* bottom_data = bottom[0]->cpu_data()";
  // const Dtype* bottom_data = bottom[0]->cpu_data();
  // if (bottom[0] == top[0]) {
  //   // In-place computation; need to store bottom data before overwriting it.
  //   // Note that this is only necessary for Backward; we could skip this if not
  //   // doing Backward, but Caffe currently provides no way of knowing whether
  //   // we'll need to do Backward at the time of the Forward call.
  //   caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),
  //              this->temp_.mutable_cpu_data());
  // }
LOG(INFO) << "debugOutput "<<" Dtype* scale_data =";


  // const Dtype* scale_data =
  //     ((bottom.size() > 1) ? bottom[1] : this->weights_quantized_[0].get())->cpu_data();
  // Dtype* top_data = top[0]->mutable_cpu_data();
  // for (int n = 0; n < this->outer_dim_; ++n) {
  //   for (int d = 0; d < this->scale_dim_; ++d) {
  //     const Dtype factor = scale_data[d];
  //     caffe_cpu_scale(this->inner_dim_, factor, bottom_data, top_data);
  //     bottom_data += this->inner_dim_;
  //     top_data += this->inner_dim_;
  //   }
  // }

  // //   // Trim layer output
  // // if (this->phase_ == TEST) {
  // //   this->QuantizeLayerOutputs_cpu(top_data, top[0]->count());
  // // }
  // LOG(INFO) << "debugOutput "<<" this->bias_layer_->Forward(this->bias_bottom_vec_, top)";
  // if (this->bias_layer_) {
  //   this->bias_layer_->Forward(this->bias_bottom_vec_, top);
  // }
  // // Trim layer output
  // if (this->phase_ == TEST) {
  //   this->QuantizeLayerOutputs_cpu(top_data, top[0]->count());
  // }
  // Trim layer input
  LOG(INFO) << "debugOutput "<<" if (this->phase_ == TEST) ";
  if (this->phase_ == TEST) {
    this->QuantizeLayerInputs_cpu(bottom[0]->mutable_cpu_data(),
        bottom[0]->count());
  }
  // Trim weights
  caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
      this->weights_quantized_[0]->mutable_cpu_data());

  LOG(INFO) << "debugOutput "<<"this->QuantizeWeights_cpu";
  //*******************************************************************************quantized bias_layer
  int rounding = this->phase_ == TEST ? this->rounding_ :
      QuantizationParameter_Rounding_STOCHASTIC;
  this->QuantizeWeights_cpu(this->weights_quantized_[0]->mutable_cpu_data(),
    this->weights_quantized_[0]->count(), rounding,this->type(),0);
  
  LOG(INFO) << "debugOutput "<<" const Dtype* bottom_data = ";
  const Dtype* bottom_data = bottom[0]->cpu_data();
  if (bottom[0] == top[0]) {
    // In-place computation; need to store bottom data before overwriting it.
    // Note that this is only necessary for Backward; we could skip this if not
    // doing Backward, but Caffe currently provides no way of knowing whether
    // we'll need to do Backward at the time of the Forward call.
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),
               this->temp_.mutable_cpu_data());
  }
  LOG(INFO) << "debugOutput "<<"  const Dtype* scale_data = ";
  const Dtype* scale_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  LOG(INFO) << "debugOutput "<<" for (int n = 0; n < this->outer_dim_; ++n) ";
  for (int n = 0; n < this->outer_dim_; ++n) {
    for (int d = 0; d < this->scale_dim_; ++d) {
      const Dtype factor = scale_data[d];
      caffe_cpu_scale(this->inner_dim_, factor, bottom_data, top_data);
      bottom_data += this->inner_dim_;
      top_data += this->inner_dim_;
    }
  }
  LOG(INFO) << "debugOutput "<<" if (this->bias_layer_) ";
  if (this->bias_layer_) {
    // Trim weights
    caffe_copy(this->weights_quantized_[this->bias_param_id_]->count(), this->blobs_[this->bias_param_id_]->cpu_data(),
       this->weights_quantized_[this->bias_param_id_]->mutable_cpu_data());

    LOG(INFO) << "debugOutput "<<"this->QuantizeWeights_cpu";

    this->QuantizeWeights_cpu(this->weights_quantized_[this->bias_param_id_]->mutable_cpu_data(),
        this->weights_quantized_[this->bias_param_id_]->count(), rounding,this->type(),0);

    this->bias_layer_->Forward(this->bias_bottom_vec_, top);
  }
  LOG(INFO) << "debugOutput "<<"  if (this->phase_ == TEST) ";
    // Trim layer output
  if (this->phase_ == TEST) {
    this->QuantizeLayerOutputs_cpu(top[0]->mutable_cpu_data(), top[0]->count());
  }
}

template <typename Dtype>
void ScaleRistrettoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->bias_layer_ &&
      this->param_propagate_down_[this->param_propagate_down_.size() - 1]) {
    this->bias_layer_->Backward(top, this->bias_propagate_down_, this->bias_bottom_vec_);
    //  copy of  quantized diff to blob[this->bias_param_id_]
    caffe_copy(this->blobs_[this->bias_param_id_]->count(), this->weights_quantized_[this->bias_param_id_]->cpu_diff(),
       this->blobs_[this->bias_param_id_]->mutable_cpu_diff());
  }
  const bool scale_param = (bottom.size() == 1);
  Blob<Dtype>* scale = scale_param ? this->blobs_[0].get() : bottom[1];
  if ((!scale_param && propagate_down[1]) ||
      (scale_param && this->param_propagate_down_[0])) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const bool in_place = (bottom[0] == top[0]);
    const Dtype* bottom_data = (in_place ? &this->temp_ : bottom[0])->cpu_data();
    // Hack: store big eltwise product in bottom[0] diff, except in the special
    // case where this layer itself does the eltwise product, in which case we
    // can store it directly in the scale diff, and we're done.
    // If we're computing in-place (and not doing eltwise computation), this
    // hack doesn't work and we store the product in temp_.
    const bool is_eltwise = (bottom[0]->count() == scale->count());
    Dtype* product = (is_eltwise ? scale->mutable_cpu_diff() :
        (in_place ? this->temp_.mutable_cpu_data() : bottom[0]->mutable_cpu_diff()));
    caffe_mul(top[0]->count(), top_diff, bottom_data, product);
    if (!is_eltwise) {
      Dtype* sum_result = NULL;
      if (this->inner_dim_ == 1) {
        sum_result = product;
      } else if (this->sum_result_.count() == 1) {
        const Dtype* sum_mult = this->sum_multiplier_.cpu_data();
        Dtype* scale_diff = scale->mutable_cpu_diff();
        if (scale_param) {
          Dtype result = caffe_cpu_dot(this->inner_dim_, product, sum_mult);
          *scale_diff += result;
        } else {
          *scale_diff = caffe_cpu_dot(this->inner_dim_, product, sum_mult);
        }
      } else {
        const Dtype* sum_mult = this->sum_multiplier_.cpu_data();
        sum_result = (this->outer_dim_ == 1) ?
            scale->mutable_cpu_diff() : this->sum_result_.mutable_cpu_data();
        caffe_cpu_gemv(CblasNoTrans, this->sum_result_.count(), this->inner_dim_,
                       Dtype(1), product, sum_mult, Dtype(0), sum_result);
      }
      if (this->outer_dim_ != 1) {
        const Dtype* sum_mult = this->sum_multiplier_.cpu_data();
        Dtype* scale_diff = scale->mutable_cpu_diff();
        if (this->scale_dim_ == 1) {
          if (scale_param) {
            Dtype result = caffe_cpu_dot(this->outer_dim_, sum_mult, sum_result);
            *scale_diff += result;
          } else {
            *scale_diff = caffe_cpu_dot(this->outer_dim_, sum_mult, sum_result);
          }
        } else {
          caffe_cpu_gemv(CblasTrans, this->outer_dim_, this->scale_dim_,
                         Dtype(1), sum_result, sum_mult, Dtype(scale_param),
                         scale_diff);
        }
      }
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* scale_data = this->weights_quantized_[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int n = 0; n < this->outer_dim_; ++n) {
      for (int d = 0; d < this->scale_dim_; ++d) {
        const Dtype factor = scale_data[d];
        caffe_cpu_scale(this->inner_dim_, factor, top_diff, bottom_diff);
        bottom_diff += this->inner_dim_;
        top_diff += this->inner_dim_;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ScaleRistrettoLayer);
#endif

INSTANTIATE_CLASS(ScaleRistrettoLayer);
REGISTER_LAYER_CLASS(ScaleRistretto);

}  // namespace caffe
