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

#ifndef SIGMOID_Adaptive_LOSS_OP_H_
#define SIGMOID_Adaptive_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SigmoidAdaptiveDistillLossOp final : public Operator<Context> {
 public:
  SigmoidAdaptiveDistillLossOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.)),
        num_classes_(OperatorBase::GetSingleArgument<int>("num_classes", 80)),
        gamma_(OperatorBase::GetSingleArgument<float>("gamma", 1.)),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 0.25)),
        beta_((OperatorBase::GetSingleArgument<float>("beta", 0))),
        ignored_label(OperatorBase::GetSingleArgument<int>("ignored_label", -1)){
    CAFFE_ENFORCE(scale_ >= 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float scale_;
  int num_classes_;
  float gamma_;
  float alpha_;
  float beta_;
  int ignored_label;
  Tensor<Context> losses_;
  Tensor<Context> counts_;
};

template <typename T, class Context>
class SigmoidAdaptiveDistillLossGradientOp final : public Operator<Context> {
 public:
  SigmoidAdaptiveDistillLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.)),
        num_classes_(OperatorBase::GetSingleArgument<int>("num_classes", 80)),
        gamma_(OperatorBase::GetSingleArgument<float>("gamma", 1.)),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 0.25)),
        beta_((OperatorBase::GetSingleArgument<float>("beta", 0))),
        ignored_label(OperatorBase::GetSingleArgument<int>("ignored_label", -1)){
    CAFFE_ENFORCE(scale_ >= 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float scale_;
  int num_classes_;
  float gamma_;
  float alpha_;
  float beta_;
  int ignored_label;
  Tensor<Context> counts_;
  Tensor<Context> weights_; // unignored weights
};

} // namespace caffe2

#endif 
