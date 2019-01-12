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

#include "sigmoid_adaptive_distillation_loss_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SigmoidAdaptiveDistillLoss, SigmoidAdaptiveDistillLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SigmoidAdaptiveDistillLossGradient,
    SigmoidAdaptiveDistillLossGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SigmoidAdaptiveDistillLoss)
    .NumInputs(4)
    .NumOutputs(1)
    .Arg(
       "scale",
       "(float) default 1.0; multiply the loss by this scale factor.")
    .Arg(
       "alpha",
       "(float) default 0.25; Focal Loss's alpha hyper-parameter.")
    .Arg(
       "gamma",
       "(float) default 1.0; Focal Loss's gamma hyper-parameter.")
    .Arg(
       "num_classes",
       "(int) default 80; number of classes (excluding background).")
    .Arg(
        "ignored_label",
        "(int) default -1; number of classes (excluding background).")
    .Arg(
        "static_focal_target",
        "(bool) default false; number of classes (excluding background).")
    .Input(
       0,
       "logits",
       "4D tensor of sigmoid inputs (called 'scores' or 'logits') with shape "
       "(N, C, H, W), where C = num_anchors * num_classes.")
    .Input(
       1,
       "labels",
       "4D tensor of labels with shape (N, num_anchors, H, W). Each entry is "
       "a class label in [0, num_classes - 1] (inclusive). The label "
       "identifies the one class that should have a sigmoid target of 1.")
    .Input(
        2,
        "GT",
        "GT")
    .Input(
       3,
       "normalizer",
       "Scalar; the loss is normalized by 1 / max(1, normalizer).")
    .Output(
       0,
       "loss",
       "Scalar loss.");

OPERATOR_SCHEMA(SigmoidAdaptiveDistillLossGradient)
    .NumInputs(5)
    .NumOutputs(1)
    .Input(
        0,
        "logits",
        "See SigmoidAdaptiveLoss.")
    .Input(
        1,
        "labels",
        "See SigmoidAdaptiveLoss.")
    .Input(
        2,
        "GT",
        "GT")
    .Input(
        3,
        "normalizer",
        "See SigmoidAdaptiveLoss.")
    .Input(
        4,
        "d_loss",
        "Gradient of forward output 0 (loss)")
    .Output(
        0,
        "d_logits",
        "Gradient of forward input 0 (logits)");

class GetSigmoidAdaptiveDistillLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  vector<OperatorDef> GetGradientDefs() override {
    vector<string> blob_names{
        {I(0), I(1), I(2),I(3), GO(0)},
    };

    return SingleGradientDef(
        "SigmoidAdaptiveDistillLossGradient", "", blob_names, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(SigmoidAdaptiveDistillLoss, GetSigmoidAdaptiveDistillLossGradient);

} // namespace caffe2
