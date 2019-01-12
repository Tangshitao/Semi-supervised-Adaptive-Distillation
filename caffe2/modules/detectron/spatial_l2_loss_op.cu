#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "spatial_l2_loss_op.h"

namespace caffe2 {

namespace {


__global__ void SpatialL2LossKernel(
    const int N, const int ignored_label, const float* X, const float* Y,float* Distance,const float *W) {
  CUDA_1D_KERNEL_LOOP(i, N) {
      if(Y[i]!=ignored_label)
        Distance[i]=(X[i]-Y[i])*(X[i]-Y[i])/(2*W[0]);
  }
}
}  // namespace

template <>
bool SpatialL2LossOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& normalizer=Input(2);
  auto* avg_loss = Output(0);

  avg_loss->Resize(vector<TIndex>());

  const float *W=normalizer.data<float>();

  Tensor<CUDAContext>distance;
  CAFFE_ENFORCE_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE_EQ(
        X.dim32(i),
        Y.dim32(i),
        "Mismatch in dimensions",
        X.dims(),
        " / ",
        Y.dims());
  }
  int N = X.size();
  distance.ResizeLike(X);

  

  SpatialL2LossKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,ignored_label, X.data<float>(), Y.data<float>(), distance.mutable_data<float>(),W);

  float* avg_loss_data=avg_loss->mutable_data<float>();
  math::Sum<float, CUDAContext>(
      distance.size(), distance.data<float>(), avg_loss_data, &context_);
  math::Scale<float, CUDAContext>(
      1, scale_, avg_loss_data, avg_loss_data, &context_);
  return true;
}

__global__ void SpatialL2LossGradientKernel(
    const int N,const int ignored_label, const float* X,const float * Y, const float* dy,float* dx,const float *W) {
  CUDA_1D_KERNEL_LOOP(i, N) {
      if(Y[i]!=ignored_label)
        dx[i]=dy[0]*(X[i]-Y[i])/W[0];
  }
}
  // namespace


template <>
bool SpatialL2LossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& normalizer=Input(2);
  auto& dtop=Input(3);
  auto* dX = Output(0);

  dX->ResizeLike(X);

  const float *W=normalizer.data<float>();

  Tensor<CUDAContext>distance;
  CAFFE_ENFORCE_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE_EQ(
        X.dim32(i),
        Y.dim32(i),
        "Mismatch in dimensions",
        X.dims(),
        " / ",
        Y.dims());
  }
  int N = X.size();

  SpatialL2LossGradientKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,ignored_label, X.data<float>(), Y.data<float>(), dtop.data<float>(),dX->mutable_data<float>(), W);
  math::Scale<float, CUDAContext>(
    dX->size(), scale_, dX->data<float>(), dX->mutable_data<float>(),
    &context_);
    return true;
}
REGISTER_CUDA_OPERATOR(SpatialL2Loss,
                       SpatialL2LossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SpatialL2LossGradient,
                       SpatialL2LossGradientOp<float, CUDAContext>);
}