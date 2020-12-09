// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void conv_3x3s1_patdnn_fp32(const float* i_data,
                            float* o_data,
                            int bs,
                            int oc,
                            int oh,
                            int ow,
                            int ic,
                            int ih,
                            int win,
                            const float* weights,
                            const float* bias,
                            int group_size,
                            std::vector<int> offset_array,
                            std::vector<int> reorder_array,
                            std::vector<int> index_array,
                            std::vector<int> stride_array,
                            const operators::ConvParam& param,
                            ARMContext* ctx);
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
