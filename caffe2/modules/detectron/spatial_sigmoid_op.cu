#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "spatial_sigmoid_op.h"

namespace caffe2 {

namespace {


__global__ void SpatialSigmoidKernel(const int N,
    const float* logits, const float* targets,float* loss) {
  CUDA_1D_KERNEL_LOOP(index, N) {
      loss[index]=-1. * logits[index] * (targets[index] - (logits[index] >= 0)) +
          logf(1 + expf(logits[index] - 2 * logits[index] * (logits[index] >= 0)));
  }
}
}  // namespace

template <>
bool SpatialSigmoidOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  
  int N = X.size();
  auto* loss = Output(0);

  loss->ResizeLike(X);


  SpatialSigmoidKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,X.data<float>(), Y.data<float>(), loss->mutable_data<float>());

  return true;
}

REGISTER_CUDA_OPERATOR(SpatialSigmoid,
                       SpatialSigmoidOp<float, CUDAContext>);

}