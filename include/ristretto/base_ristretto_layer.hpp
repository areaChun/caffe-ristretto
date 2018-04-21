#ifndef CAFFE_BASE_RISTRETTO_LAYER_HPP_
#define CAFFE_BASE_RISTRETTO_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides quantization methods used by other Ristretto layers.
 */
template <typename Dtype>
class BaseRistrettoLayer{
 public:
  explicit BaseRistrettoLayer();
 protected:
  void QuantizeLayerOutputs_cpu(Dtype* data, const int count);
  void QuantizeLayerInputs_cpu(Dtype* data, const int count);
  void QuantizeLayerOutputs_gpu(Dtype* data, const int count);
  void QuantizeLayerInputs_gpu(Dtype* data, const int count);
  void QuantizeWeights_cpu(Dtype* weight, const int count, const int rounding,
    const char* type_name, const int blob_dim);
  void QuantizeWeights_gpu(Dtype* weight, const int count, const int rounding,
    const char* type_name, const int blob_dim);
  void QuantizeWeights_cpu(vector<shared_ptr<Blob<Dtype> > > weights_quantized,
      const int rounding, const bool bias_term = true);
  void QuantizeWeights_gpu(vector<shared_ptr<Blob<Dtype> > > weights_quantized,
      const int rounding, const bool bias_term = true);
  /**
   * @brief Trim data to fixed point.
   * @param fl The number of bits in the fractional part.
   */
  void Trim2FixedPoint_cpu(Dtype* data, const int cnt, const int bit_width,
      const int rounding, int fl);
  void Trim2FixedPoint_gpu(Dtype* data, const int cnt, const int bit_width,
      const int rounding, int fl);
  /**
   * @brief Trim data to minifloat.
   * @param bw_mant The number of bits used to represent the mantissa.
   * @param bw_exp The number of bits used to represent the exponent.
   */
  void Trim2MiniFloat_cpu(Dtype* data, const int cnt, const int bw_mant,
      const int bw_exp, const int rounding);
  void Trim2MiniFloat_gpu(Dtype* data, const int cnt, const int bw_mant,
      const int bw_exp, const int rounding);
  /**
   * @brief Trim data to integer-power-of-two numbers.
   * @param min_exp The smallest quantized value is 2^min_exp.
   * @param min_exp The largest quantized value is 2^max_exp.
   */
  void Trim2IntegerPowerOf2_cpu(Dtype* data, const int cnt, const int min_exp,
      const int max_exp, const int rounding);
  void Trim2IntegerPowerOf2_gpu(Dtype* data, const int cnt, const int min_exp,
      const int max_exp, const int rounding);
  /**
   * @brief Generate random number in [0,1) range.
   */
  inline double RandUniform_cpu();
  // The number of bits used for dynamic fixed point parameters and layer
  // activations. bw_params_ is the param bit length of Conv && Fc && Scale
  int bw_params_, bw_layer_in_, bw_layer_out_;
  // bw of Batch Norm layer
  int bw_mean_, bw_var_, bw_mov_;
  // The fractional length of dynamic fixed point numbers. fl_params_ is the param bit length of Conv && Fc && Scale
  int fl_params_, fl_layer_in_, fl_layer_out_;
    // fl of Batch Norm layer
  int fl_mean_, fl_var_, fl_mov_;
  // The number of bits used to represent mantissa and exponent of minifloat
  // numbers.
  int fp_mant_, fp_exp_;
  // Integer-power-of-two numbers are in range +/- [2^min_exp, 2^max_exp].
  int pow_2_min_exp_, pow_2_max_exp_;
  // The rounding mode for quantization and the quantization scheme.
  int rounding_, precision_;

  vector<vector<int > > bw_params_vec_, fl_params_vec_;
  // For parameter layers: reduced word with parameters.
  vector<shared_ptr<Blob<Dtype> > > weights_quantized_;
};

/**
 * @brief Convolutional layer with quantized layer parameters and activations.
 */
template <typename Dtype>
class ConvolutionRistrettoLayer : public ConvolutionLayer<Dtype>,
      public BaseRistrettoLayer<Dtype> {
 public:
  explicit ConvolutionRistrettoLayer(const LayerParameter& param);
  virtual inline const char* type() const { return "ConvolutionRistretto"; }

 protected:
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

/**
 * @brief Deconvolutional layer with quantized layer parameters and activations.
 */
template <typename Dtype>
class DeconvolutionRistrettoLayer : public DeconvolutionLayer<Dtype>,
      public BaseRistrettoLayer<Dtype> {
 public:
  explicit DeconvolutionRistrettoLayer(const LayerParameter& param);

  virtual inline const char* type() const { return "DeconvolutionRistretto"; }

 protected:
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

/**
 * @brief Inner product (fully connected) layer with quantized layer parameters
 * and activations.
 */
template <typename Dtype>
class FcRistrettoLayer : public InnerProductLayer<Dtype>,
      public BaseRistrettoLayer<Dtype>{
 public:
  explicit FcRistrettoLayer(const LayerParameter& param);
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "FcRistretto"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

/**
 * @brief Local response normalization (LRN) layer with minifloat layer inputs,
 * intermediate results and outputs.
 */
template <typename Dtype>
class LRNRistrettoLayer : public LRNLayer<Dtype>,
      public BaseRistrettoLayer<Dtype>{
 public:
  explicit LRNRistrettoLayer(const LayerParameter& param);
  virtual inline const char* type() const { return "LRNRistretto"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
};

/**
 * @brief batch normalization (BatchNorm) layer with minifloat layer inputs,
 * intermediate results and outputs.
 */

template <typename Dtype>
class BatchNormRistrettoLayer : public BatchNormLayer<Dtype>, 
      public BaseRistrettoLayer<Dtype>{
 public:
  explicit BatchNormRistrettoLayer(const LayerParameter& param);
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "BatchNormRistretto"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};

/**
 * @brief scale (Scale) layer with minifloat layer inputs,
 * intermediate results and outputs.
 */

template <typename Dtype>
class ScaleRistrettoLayer: public ScaleLayer<Dtype>, 
      public BaseRistrettoLayer<Dtype>{
 public:
  explicit ScaleRistrettoLayer(const LayerParameter& param);
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ScaleRistretto"; }
  // Scale
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * In the below shape specifications, @f$ i @f$ denotes the value of the
   * `axis` field given by `this->layer_param_.scale_param().axis()`, after
   * canonicalization (i.e., conversion from negative to positive index,
   * if applicable).
   *
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the first factor @f$ x @f$
   *   -# @f$ (d_i \times ... \times d_j) @f$
   *      the second factor @f$ y @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the product @f$ z = x y @f$ computed after "broadcasting" y.
   *      Equivalent to tiling @f$ y @f$ to have the same shape as @f$ x @f$,
   *      then computing the elementwise product.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};

/**
 * @brief bias (Bias) layer with minifloat layer inputs,
 * intermediate results and outputs.
 */

template <typename Dtype>
class BiasRistrettoLayer : public BiasLayer<Dtype>, 
      public BaseRistrettoLayer<Dtype>{
 public:
  explicit BiasRistrettoLayer(const LayerParameter& param);
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // void Reshape(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BiasRistretto"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  Blob<Dtype> bias_multiplier_;
  int outer_dim_, bias_dim_, inner_dim_, dim_;
};


}  // namespace caffe

#endif  // CAFFE_BASE_RISTRETTO_LAYER_HPP_
