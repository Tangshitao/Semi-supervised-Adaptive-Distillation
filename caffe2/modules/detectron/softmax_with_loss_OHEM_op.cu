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

#include <cfloat>
#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/softmax_op.h"
#include "softmax_with_loss_OHEM_op.h"
#include "caffe2/operators/softmax_with_loss_op.h"
namespace caffe2 {

namespace {

__global__ void LabelCrossEntropyOHEMKernel(
    const int N,
    const int D,
    const float* logPdata,
    const int* labeldata,
    float* Ydata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    CUDA_KERNEL_ASSERT(labeldata[i] >= 0 && labeldata[i] < D);
    Ydata[i] = -logPdata[i * D + labeldata[i]];
  }
}

__global__ void SoftmaxNormalizeLogsKernel(
    const int nthreads,
    const int D,
    const float* logits,
    const float* rowmax,
    const float* scales,
    float* out_log) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / D;
    out_log[index] = logits[index] - rowmax[n] - logf(max(scales[n], FLT_MIN));
  }
}

__global__ void SoftmaxNormalizeKernel(
    const int nthreads,
    const int D,
    const float* probs,
    const float* scales,
    float* out) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / D;
    out[index] = probs[index] / scales[n];
  }
}

void Softmax(
    const int N,
    const int D,
    const float* logits,
    const float* sum_multiplier,
    float* scales,
    float* rowmax,
    float* probs,
    bool log_softmax,
    CUDAContext* context) {
  const int size = N * D;

  math::RowwiseMax<float, CUDAContext>(N, D, logits, rowmax, context);
  // Put the intermediate result X - max(X) into Y
  context->Copy<float, CUDAContext, CUDAContext>(size, logits, probs);
  // Subtract the scale
  math::Gemm<float, CUDAContext>(
      CblasNoTrans,
      CblasNoTrans,
      N,
      D,
      1,
      -1,
      rowmax,
      sum_multiplier,
      1,
      probs,
      context);
  // Exponentiation
  math::Exp<float, CUDAContext>(size, probs, probs, context);
  // Sum exponentiated values
  math::Gemv<float, CUDAContext>(CblasNoTrans, N, D, 1, probs, sum_multiplier,
                                 0, scales, context);
  // Normalize
  if (!log_softmax) {
    SoftmaxNormalizeKernel<<<
        CAFFE_GET_BLOCKS(size),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context->cuda_stream()>>>(size, D, probs, scales, probs);
  } else {
    SoftmaxNormalizeLogsKernel<<<
        CAFFE_GET_BLOCKS(size),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context->cuda_stream()>>>(size, D, logits, rowmax, scales, probs);
  }
}

}

template<>
bool SoftmaxWithLossOHEMOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Logits
  auto& T = Input(1);  // Labels / targets
  auto* P = Output(0); // Probabilities from softmax
  auto* L = Output(1);

  const auto canonical_axis = X.canonical_axis_index(axis_);
  int N, D;
  N = X.size_to_dim(canonical_axis); // batch size
  D = X.size_from_dim(canonical_axis);
  P->ResizeLike(X);


  if (T.ndim() == canonical_axis) {
    CAFFE_ENFORCE_EQ(T.size(), N);
  } else {
    CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), 1);
  }
  


  if (losses_.size() != N) {
    losses_.Resize(N);
  }
  if (rowmax_.size() != N) {
    rowmax_.Resize(N);
  }
  L->Resize(N);
  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CUDAContext>(
        D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }
  Softmax(
      N,
      D,
      X.data<float>(),
      sum_multiplier_.data<float>(),
      losses_.mutable_data<float>(),
      rowmax_.mutable_data<float>(),
      P->mutable_data<float>(),
      true, // logarithmic output
      &context_);
  // Compute label xent loss per example

  LabelCrossEntropyOHEMKernel<<<
    CAFFE_GET_BLOCKS(N),
    CAFFE_CUDA_NUM_THREADS,
    0,
    context_.cuda_stream()>>>(
    N,
    D,
    P->data<float>(),
    T.data<int>(),
    L->mutable_data<float>());
    // Since we had logarithmic output, we need to exponentiate
    // them again.
    math::Exp<float, CUDAContext>(
        N * D, P->data<float>(), P->mutable_data<float>(), &context_);

  return true;
}
REGISTER_CUDA_OPERATOR(SoftmaxWithLossOHEM,
                       SoftmaxWithLossOHEMOp<float, CUDAContext>);
}