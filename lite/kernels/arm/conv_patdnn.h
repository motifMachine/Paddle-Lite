// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cmath>
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))

template <PrecisionType Ptype, PrecisionType OutType>
inline bool direct_conv_trans_weights(
    const Tensor* win,
    Tensor* wout,
    const Tensor* bin,
    Tensor* bout,
    int stride,
    const std::vector<float>& w_scale,
    float in_scale,
    float out_scale,
    std::vector<float>& merge_scale,  // NOLINT
    float* relu_clipped_coef) {
  constexpr int cblock = 4;
  int oc = win->dims()[0];
  int ic = win->dims()[1];
  int kh = win->dims()[2];
  int kw = win->dims()[3];
  int cround = ROUNDUP(oc, cblock);
  wout->Resize({cround, ic, kh, kw});
  auto w_in_data = win->data<float>();
  auto transed_w_data = wout->mutable_data<float>();
  lite::arm::math::conv_trans_weights_numc(
      w_in_data, transed_w_data, oc, ic, cblock, kh * kw);
  return false;
}

/// only support 3x3s1
template <PrecisionType Ptype, PrecisionType OutType>
class PatDNNConv : public KernelLite<TARGET(kARM), Ptype> {
 public:
  PatDNNConv() = default;
  ~PatDNNConv() {}

  virtual void PrepareForRun() {
    auto& param = this->template Param<param_t>();
    auto& ctx = this->ctx_->template As<ARMContext>();

    auto x_dims = param.x->dims();
    auto w_dims = param.filter->dims();
    auto o_dims = param.output->dims();

    int ic = x_dims[1];
    int oc = o_dims[1];
    int sw = param.strides[1];
    int kw = w_dims[3];
    int kh = w_dims[2];
    CHECK(sw == 1 || sw == 2)
        << "patdnn conv only support conv3x3s1";
    CHECK(kw == 3 && kh == 3)
        << "patdnn conv only support conv3x3s1";
    flag_trans_bias_ = direct_conv_trans_weights<Ptype, OutType>(
        param.filter,
        &weights_,
        param.bias,
        &bias_,
        sw,
        param.weight_scale,
        param.input_scale,
        param.output_scale,
        w_scale_,
        &param.activation_param.Relu_clipped_coef);
  }

  virtual void Run();

  /// todo, support inplace weights transform
 protected:
  Tensor weights_;
  Tensor bias_;
  bool flag_trans_weights_{false};
  bool flag_trans_bias_{false};
  std::vector<float> w_scale_;

 private:
  using param_t = operators::ConvParam;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
