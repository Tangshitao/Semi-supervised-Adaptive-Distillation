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

#include "caffe2/core/context_gpu.h"
#include "smooth_l1_loss_OHEM_op.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void SmoothL1Kernel(
    const int n, const T* in, T* out, T beta) {
  // f(x) = 0.5 * x^2 / beta      if |x| < beta
  //        |x| - 0.5 * beta      otherwise
  CUDA_1D_KERNEL_LOOP(index, n) {
    T val = in[index];
    T abs_val = abs(val);
    if (abs_val < beta) {
      out[index] = 0.5 * val * val / beta;
    } else {
      out[index] = abs_val - 0.5 * beta;
    }
  }
}
template <typename T>
__global__ void kernel_channel_sum(const int num, const int channels,
 const T* data, T* channel_sum) {
    CUDA_1D_KERNEL_LOOP(index, num) {
        int n = index;
        T sum = 0;
        for (int c = 0; c < channels; ++c) {
        sum += data[n * channels + c];
        }
        channel_sum[index] = sum;
    }
}
} // namespace



template<>
bool SmoothL1LossOHEMOp<float, CUDAContext>::RunOnDevice() {
  auto& Y_hat     = Input(0);
  auto& Y         = Input(1);
  auto& alpha_in  = Input(2);
  auto& alpha_out = Input(3);
  auto* L  = Output(0);

  int N = Y.dim32(0);
  int channel_num=Y.dim32(1);
  // Require the same number of elements along axis 0 (batch size), but
  // otherwise don't care about the shape (just the number of elements)
  CAFFE_ENFORCE_EQ(Y_hat.dim32(0), Y.dim32(0),
      "Y_hat and Y must have the same number of elements along axis 0");
  CAFFE_ENFORCE_EQ(Y_hat.size(), Y.size(),
      "Y_hat and Y must have the same number of elements");
  CAFFE_ENFORCE_EQ(Y_hat.size(), alpha_in.size());
  CAFFE_ENFORCE_EQ(Y_hat.size(), alpha_out.size());

  buff_.ResizeLike(Y);
  L->Resize(N);

  // Difference
  // d := y_hat - y
  math::Sub<float, CUDAContext>(
      Y.size(), Y_hat.data<float>(), Y.data<float>(),
      buff_.mutable_data<float>(), &context_);
  // Element-wise weighted difference (can be used to ignore or reweight
  // specific components)
  // d := alpha_in * (y_hat - y)
  math::Mul<float, CUDAContext>(
      buff_.size(), buff_.data<float>(), alpha_in.data<float>(),
      buff_.mutable_data<float>(), &context_);

  // Element-wise smooth l1 loss
  // l := SmoothL1(alpha_in * (y_hat - y))
  SmoothL1Kernel<float>
  <<<CAFFE_GET_BLOCKS(buff_.size()),
     CAFFE_CUDA_NUM_THREADS,
     0,
     context_.cuda_stream()>>>(
          buff_.size(), buff_.data<float>(), buff_.mutable_data<float>(),
          beta_);

  // Element-wise weighted smooth l1 loss (can be used to specify a per-element
  // loss weight)
  // l := alpha_out * SmoothL1(alpha_in * (y_hat - y))
  math::Mul<float, CUDAContext>(
      buff_.size(), buff_.data<float>(), alpha_out.data<float>(),
      buff_.mutable_data<float>(), &context_);
  
  kernel_channel_sum<float>
  <<<CAFFE_GET_BLOCKS(buff_.size()),
     CAFFE_CUDA_NUM_THREADS,
     0,
     context_.cuda_stream()>>>(
         N,channel_num,buff_.data<float>(),L->mutable_data<float>()
     );
  return true;
}


REGISTER_CUDA_OPERATOR(SmoothL1LossOHEM,
                       SmoothL1LossOHEMOp<float, CUDAContext>);

}  // namespace caffe2
