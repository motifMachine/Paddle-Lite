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

#include <stdlib.h>
#include <string.h>
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

inline bool init_test_sparse_format(std::vector<int>* offset_array,
                                    std::vector<int>* reorder_array,
                                    std::vector<int>* index_array,
                                    std::vector<int>* stride_array,
                                    int oc,
                                    int ic,
                                    int GroupSize) {
  int num_group = (oc + GroupSize - 1) /
                  GroupSize;  // 共有num_group个group，如果不能整除，向上取整
  // init offset array
  (*offset_array).push_back(0);
  int kernels_per_group = oc / num_group;
  int kernels = 0;
  for (int i = 0; i < num_group - 1; i++) {
    for (int j = 0; j < GroupSize; j++) {
      kernels += kernels_per_group;
      (*offset_array).push_back(kernels);
    }
    kernels_per_group += oc / num_group;
  }
  for (int i = (num_group - 1) * GroupSize; i < oc; i++) {
    kernels += kernels_per_group;
    (*offset_array).push_back(kernels);
  }

  // init reorder array
  for (int i = 0; i < oc; i++) {
    (*reorder_array).push_back(i);
  }

  // init index array
  bool* flag;
  flag = reinterpret_cast<bool*> malloc(sizeof(bool) * ic);
  memset(flag, false, sizeof(bool) * ic);
  unsigned int seed = 1;
  for (int i = 0; i < oc; i++) {
    for (int j = 0; j < (*offset_array)[i + 1] - (*offset_array)[i]; j++) {
      int index = rand_r(&seed) % ic;
      while (flag[index]) {
        index = rand_r(&seed) % ic;
      }
      flag[index] = true;
      (*index_array).push_back(index);
    }
    memset(flag, false, sizeof(bool) * ic);
  }

  // init stride array
  for (int i = 0; i < oc; i++) {
    for (int j = 0; j < (*offset_array)[i + 1] - (*offset_array)[i]; j++) {
      (*stride_array).push_back(0);
    }
    (*stride_array).push_back((*offset_array)[i + 1] - (*offset_array)[i]);
  }

  // compute the memory offset replace weight offset
  for (int i = 0; (*offset_array).size(); i++) {
    (*offset_array).data()[i] = (*offset_array).data()[i] * 4;
  }
}

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
    CHECK(sw == 1 || sw == 2) << "patdnn conv only support conv3x3s1";
    CHECK(kw == 3 && kh == 3) << "patdnn conv only support conv3x3s1";
    init_test_sparse_format(offset_array,
                            reorder_array,
                            index_array,
                            stride_array,
                            oc,
                            ic,
                            group_size);
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

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
  }

  std::string kernel_func_name_{"NotImplForConvDirect"};
#endif

  /// todo, support inplace weights transform
 protected:
  Tensor weights_;
  Tensor bias_;
  bool flag_trans_weights_{false};
  bool flag_trans_bias_{false};
  std::vector<float> w_scale_;
  // For patdnn sparse conv
  // The offset of kernels in each filter
  std::vector<int> offset_array;
  // The index records the location of each filter in layer before reorder.
  std::vector<int> reorder_array;
  // The index records the location of each kernel in filter berfore reoder.
  std::vector<int> index_array;
  // The stride of each group kernel belong to different sparse pattern.
  std::vector<int> stride_array;
  // GroupSize
  int group_size = 4;

 private:
  using param_t = operators::ConvParam;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
