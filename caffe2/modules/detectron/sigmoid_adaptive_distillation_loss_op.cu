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
#include<iostream>

#include "caffe2/core/context_gpu.h"
#include "sigmoid_adaptive_distillation_loss_op.h"


namespace caffe2 {

namespace {

__global__ void SigmoidAdaptiveDistillLossKernel(
    const int N, const int D, const int H, const int W,const int ignored_label, const float* logits,
    const float* targets, const int* gt,const float* weight_pos,
    const float gamma, const float alpha,const float beta,
    const int num_classes, float* losses) { 
  CUDA_1D_KERNEL_LOOP(i, N * D * H * W) {

    int x = i % W;
    int y = (i / W) % H;
    int c = (i / (W * H)) % D;  // channel, here D is channel dim in input NxDxHxW
    int n = i / (W * H * D);    // n in NxDxHxW

    int A = D / num_classes;   // num_anchors = A
    int a = c / num_classes;   // current anchor out of A anchors in D = A * num_cls
    int t = gt[n * (H * W * A) + a * (H * W) + y * W + x];   // target

    // check whether the class is true class or not.
    // The target classes are in range 1 - 81 and the d is in range 0-80
    // because we predict A*80 dim, so for comparison purpose, compare t and (d+1)
    

    float Np = max(weight_pos[0], 1.0);
    float zn = (1.0 - alpha) / Np;
    float zp = alpha / Np;

    // p = 1. / 1. + expf(-x)
    float pt= targets[i];
    float p = 1. / (1. + exp(-logits[i]));


    float D_loss=-1.*logits[i]*(pt-(logits[i]>=0))+logf(max(FLT_MIN,1+expf(logits[i]-2*logits[i]*(logits[i]>=0))))
    +beta*(pt*logf(pt)+(1-pt)*logf(1-pt));

    float adaptive_target=1-expf(-D_loss);

    losses[i] = -powf(adaptive_target,gamma)*(pt*logf(max(FLT_MIN,p))*zp+(1-pt)*(-1.*logits[i]*(logits[i]>=0) \
    -logf(1.+expf(logits[i]-2.*logits[i]*(logits[i]>=0))))*zn)*(t!=ignored_label);

  }
}

__global__ void SigmoidAdaptiveDistillLossGradientKernel(
    const int N, const int D, const int H, const int W, const int ignored_label, const float* logits,
    const float* targets,const int *gt, float* dX_data, const float* weight_pos,
    const float gamma, const float alpha, const float beta,const int num_classes,
    const float* avg_loss) {
  CUDA_1D_KERNEL_LOOP(i, N * D * H * W) {
      float a_loss = avg_loss[0];
      int x = i % W;
      int y = (i / W) % H;
      int c = (i / (W * H)) % D;
      int n = i / (W * H * D);

      int A = D / num_classes;   // num_anchors
      int a = c / num_classes;   // current anchor
      
      int t = gt[n * (H * W * A) + a * (H * W) + y * W + x];
      
      
      float Np = max(weight_pos[0], 1.0);
      float pt= targets[i];
      float p = 1. / (1. + expf(-logits[i]));


      float DL=-1.*logits[i]*(pt-(logits[i]>=0))+logf(1+expf(logits[i]-2*logits[i]*(logits[i]>=0)))
      +beta*(pt*logf(pt)+(1-pt)*logf(1-pt)); //-(pt*log(max(p,FLT_MIN))+(1-pt)*log(max(1-p,FLT_MIN)));
      float expDL=expf(-DL);
      float adaptive_target=1-expDL;

      float DLoss=alpha*pt*logf(max(FLT_MIN,p))+(1-alpha)*(1-pt)*(-1.*logits[i]*(logits[i]>=0)-logf(1.+expf(logits[i]-2.*logits[i]*(logits[i]>=0))));
      dX_data[i]=-(-(pt-p)*gamma*powf(adaptive_target,gamma-1)*expDL*DLoss \
        +powf(adaptive_target,gamma)*(alpha*(pt-p)-(1-2*alpha)*(1-pt)*p))*a_loss*(t!=ignored_label);
        
      
      dX_data[i]/=Np;

  }
}
} // namespace

template<>
bool SigmoidAdaptiveDistillLossOp<float, CUDAContext>::RunOnDevice() {
  // Input logits, for example: N x (A * 80) x H x W in cls-agnostic
  auto& X = Input(0);
  // Target, for example: N x A x H x W
  auto& T = Input(1);
  auto& G= Input(2);
  // Number of positive examples: scalar
  auto& wp = Input(3);
  // output avg Sigmoid adaptive loss as mentioned in RetinaNet paper
  auto* avg_loss = Output(0);

  int N = X.dim32(0);
  int D = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);

  avg_loss->Resize(vector<TIndex>());
  losses_.ResizeLike(X);
  float* avg_loss_data = avg_loss->mutable_data<float>();

  SigmoidAdaptiveDistillLossKernel<<<CAFFE_GET_BLOCKS(X.size()),
          CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      N, D, H, W,ignored_label, X.data<float>(), T.data<float>(), G.data<int>(),
      wp.data<float>(), gamma_, alpha_,beta_, num_classes_,
      losses_.mutable_data<float>());

  math::Sum<float, CUDAContext>(
      losses_.size(), losses_.data<float>(), avg_loss_data, &context_);
  math::Scale<float, CUDAContext>(
      1, scale_, avg_loss_data, avg_loss_data, &context_);

  return true;
}


template<>
bool SigmoidAdaptiveDistillLossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& T = Input(1);
  auto& G = Input(2);
  auto& wp = Input(3);
  auto& d_avg_loss = Input(InputSize() - 1);
  auto* dX = Output(0);

  // get input shape
  int N = X.dim32(0);
  int D = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);

  dX->ResizeLike(X);

  SigmoidAdaptiveDistillLossGradientKernel<<<CAFFE_GET_BLOCKS(X.size()),
          CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      N, D, H, W, ignored_label,X.data<float>(), T.data<float>(),G.data<int>(), dX->mutable_data<float>(),
      wp.data<float>(), gamma_, alpha_,beta_, num_classes_,
      d_avg_loss.data<float>());

  math::Scale<float, CUDAContext>(
    dX->size(), scale_, dX->data<float>(), dX->mutable_data<float>(), &context_);

  return true;
}


REGISTER_CUDA_OPERATOR(SigmoidAdaptiveDistillLoss,
                       SigmoidAdaptiveDistillLossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SigmoidAdaptiveDistillLossGradient,
                       SigmoidAdaptiveDistillLossGradientOp<float, CUDAContext>);
} // namespace caffe2
