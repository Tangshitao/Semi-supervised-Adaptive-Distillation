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

#include "softmax_with_loss_OHEM_op.h"
#include "caffe2/operators/softmax_shared.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SoftmaxWithLossOHEM, SoftmaxWithLossOHEMOp<float, CPUContext>);

// Input: X (logits), T (labels); Output: P (probs), Y
OPERATOR_SCHEMA(SoftmaxWithLossOHEM)
    .NumInputs(2)
    .NumOutputs(2)
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
          const int num_classes =
              size_from_dim_(canonical_axis, GetDimsVector(logits));

          out[0].set_data_type(logits.data_type());
          out[0].add_dims(batch_size);
          out[0].add_dims(num_classes);
          out[1].add_dims(batch_size);

          return out;
        })
    .SetDoc(R"DOC(
Combined Softmax and Cross-Entropy loss operator.
The operator computes the softmax normalized values for each layer in the batch
of the given input, after which cross-entropy loss is computed. This operator is
numerically more stable than separate Softmax and CrossEntropy ops.
The inputs are a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions) and tensor of labels (ground truth).
Output is tensor with the probability for each label for each example (N x D)
and averaged loss (scalar).
Use parameter label_prob=1 to enable inputting labels as a probability
distribution.
Optional third input blob can be used to weight the samples for the loss.
)DOC")
    .Input(0, "logits", "Unscaled log probabilities")
    .Input(1, "labels", "Ground truth")
    .Output(0, "prob", "softmax prob")
    .Output(1, "loss", "softmax loss");
    
// Input: X, T, P, dY; Output: dX

#define DONT_CARE (-1)

template <>
bool SoftmaxWithLossOHEMOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0); // Logits
  auto& T = Input(1); // Labels / targets
  auto* P = Output(0); // Probabilities from softmax
  auto* L = Output(1);

  const auto canonical_axis = X.canonical_axis_index(axis_);
  int N, D;
  N = X.size_to_dim(canonical_axis); // batch size
  D = X.size_from_dim(canonical_axis);
  P->ResizeLike(X);

  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CPUContext>(
        D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }

  float* Pdata = P->mutable_data<float>();

  if (T.ndim() == canonical_axis) {
    CAFFE_ENFORCE_EQ(T.size(), N);
  } else {
    CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), 1);
  }
  

  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CPUContext>(
        D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }

  rowmax_.Resize(N);
  losses_.Resize(N);

  SoftmaxCPU(
      context_,
      N,
      D,
      X.data<float>(),
      Pdata,
      losses_.mutable_data<float>(),
      sum_multiplier_.data<float>(),
      true,
      rowmax_.mutable_data<float>());

  // Then compute cross entropy


  const int* label_data = T.data<int>();
  const float* Xdata = X.data<float>();
  L->Resize(N);
  float* loss_data=L->mutable_data<float>();

  for (int i = 0; i < N; ++i) {
    CAFFE_ENFORCE(
        label_data[i] < D && label_data[i] >= 0,
        "Label seems incorrect: label value larger than number of classes: ",
        label_data[i],
        " vs ",
        D);
    float l = -Pdata[i * D + label_data[i]];
    loss_data[i]=l;
  }
  math::Exp(N * D, Pdata, Pdata, &context_);
  return true;
}
}