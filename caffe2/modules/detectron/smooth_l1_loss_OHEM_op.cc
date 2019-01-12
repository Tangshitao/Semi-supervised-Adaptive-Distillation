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

#include "smooth_l1_loss_OHEM_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SmoothL1LossOHEM, SmoothL1LossOHEMOp<float, CPUContext>);

OPERATOR_SCHEMA(SmoothL1LossOHEM)
    .NumInputs(4)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          ArgumentHelper helper(def);
          auto axis = helper.GetSingleArgument<int32_t>("axis", 1);

          vector<TensorShape> out(2);

          auto logits = in[0]; // Tensor with Shape [batch_size, num_classes]
          auto labels = in[1]; // Tensor with shape [batch_size, ]
          const auto canonical_axis =
              canonical_axis_index_(axis, logits.dims().size());
          const int batch_size =
              size_to_dim_(canonical_axis, GetDimsVector(logits));


          out[0].set_data_type(logits.data_type());
          out[0].add_dims(batch_size);

          return out;
        })
    .SetDoc(R"DOC(
Smooth L1 Loss is a minor variation of Huber loss in which the point of
transition between L2 loss and L1 loss is adjustable by a hyper-parameter beta:

  SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                |x| - 0.5 * beta      otherwise.

SmoothL1 is used in Fast R-CNN and decendants as the loss function for bounding
box regression.

The loss computed by this op has a flexible form:

  scale / N * sum_i alpha_out[i] * SmoothL1(alpha_in[i] * (y_hat[i] - y[i])).

The weights alpha_in and alpha_out are called the "inside" and "outside"
weights, respectively. The inside weights are typically set to either 0 or 1 to
implement ignoring (when 0) certain samples. The outside weights can be used
to implement a per-sample loss weight. The overall loss is scaled by scale / N,
where N is the number of batch elements in the input predictions.
)DOC")
    .Arg(
        "beta",
        "(float) default 1.0; L2 to L1 transition point.")
    .Arg(
        "scale",
        "(float) default 1.0; multiply the loss by this scale factor.")
    .Input(
        0,
        "Y_hat",
        "Tensor of predictions (at least 1D).")
    .Input(
        1,
        "Y",
        "Tensor of labels with the same shape as Y_hat.")
    .Input(
        2,
        "alpha_in",
        "Tensor of inside weights with the same shape as Y.")
    .Input(
        3,
        "alpha_out",
        "Tensor of outside weights with the same shape as Y.")
    .Output(
        0,
        "loss",
        "Scalar loss.");


} // namespace caffe2
