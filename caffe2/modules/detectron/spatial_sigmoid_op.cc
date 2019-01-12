/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "spatial_sigmoid_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SpatialSigmoid, SpatialSigmoidOp<float, CPUContext>);

OPERATOR_SCHEMA(SpatialSigmoid)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Nearest neighbor upsampling operation. Implementation taken from THCUNN.
)DOC")
    .Input(
        0,
        "X",
        "4D feature map input of shape (N, C, H, W).")
    .Input(
        1,
        "Y",
        "4D feature map input of shape (N, C, H, W)."
    )
    .Output(
        0,
        "Y",
        "4D feature map of shape (N, C, scale * H, scale * W); Values are "
        "neareast neighbor samples from X.");

} // namespace caffe2
