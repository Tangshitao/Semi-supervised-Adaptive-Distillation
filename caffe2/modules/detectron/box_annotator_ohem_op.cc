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

#include "box_annotator_ohem_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(BoxAnnotatorOHEM, BoxAnnotatorOHEMOp<float, CPUContext>);

OPERATOR_SCHEMA(BoxAnnotatorOHEM)
    .NumInputs(3)
    .NumOutputs(2)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          ArgumentHelper helper(def);
          auto axis = helper.GetSingleArgument<int32_t>("axis", 1);

          vector<TensorShape> out(2);

          auto weights = in[2]; // Tensor with Shape [batch_size, num_classes]

          const auto canonical_axis =
              canonical_axis_index_(axis, weights.dims().size());
          const int batch_size =
              size_to_dim_(canonical_axis, GetDimsVector(weights));
          const int bbox_dim =
              size_from_dim_(canonical_axis, GetDimsVector(weights));

          out[0].set_data_type(weights.data_type());
          out[0].add_dims(batch_size);
          out[1].set_data_type(weights.data_type());
          out[1].add_dims(batch_size);
          out[1].add_dims(bbox_dim);

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
        "roi_per_img",
        "(int) default 128")
    .Input(
        0,
        "rois",
        "Tensor of predictions (at least 1D).")
    .Input(
        1,
        "per_roi_loss",
        "Tensor of labels with the same shape as Y_hat.")
    .Input(
        2,
        "weights",
        "Tensor of outside weights with the same shape as Y.")
    .Output(
        0,
        "label_weights",
        "label weights")
    .Output(
        1,
        "bbox_weights",
        "bbox weights");


} // namespace caffe2
