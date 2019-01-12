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

#ifndef BOX_ANNOTATOR_OHEM_OP_H_
#define BOX_ANNOTATOR_OHEM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class BoxAnnotatorOHEMOp final : public Operator<Context> {
 public:
    BoxAnnotatorOHEMOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
      roi_per_img_(OperatorBase::GetSingleArgument<int>("roi_per_img", 128)) {
      CAFFE_ENFORCE(roi_per_img_ > 0);
    }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float roi_per_img_; // Transition point from L1 to L2 loss
  
  Tensor<Context> buff_; // Buffer for element-wise differences
};

} // namespace caffe2

#endif // SMOOTH_L1_LOSS_OP_H_
