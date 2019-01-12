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

#include "pow_sum_op.h"
#include "caffe2/operators/softmax_shared.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    PowSum,
    PowSumOp<float, CPUContext>);

OPERATOR_SCHEMA(PowSum)
    .NumInputs(1,INT_MAX)
    .NumOutputs(1)
    .Arg(
        "power",
        "power"
    )
    .Output(
        0,
        "probabilities",
        "4D tensor of softmax probabilities with shape (N, C, H, W), where "
        "C = num_anchors * num_classes, and softmax was applied to each of the "
        "num_anchors groups; within a group the num_classes values sum to 1.");

}