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

#include "caffe2/core/context_gpu.h"
#include "pow_sum_op.h"

namespace caffe2 {


template <>
bool PowSumOp<float, CUDAContext>::RunOnDevice() {
  auto* res=Output(0); 
  //Tensor<CUDAContext> _buff;
  //Tensor<CUDAContext> _buff_sum;
  res->Resize(vector<TIndex>());
  _buff_sum.Resize(vector<TIndex>());
  math::Set<float,CUDAContext>(1,0,res->mutable_data<float>(),&context_);

  for(int i=0;i<InputSize();i++){
      auto &in=Input(i);
      _buff.ResizeLike(in);
      math::Powx<float,CUDAContext>(in.size(),in.data<float>(),power,_buff.mutable_data<float>(),&context_);
      math::Sum<float,CUDAContext>(_buff.size(),_buff.data<float>(),_buff_sum.mutable_data<float>(),&context_);
      math::Add<float,CUDAContext>(1,res->data<float>(),_buff_sum.data<float>(),res->mutable_data<float>(),&context_);
  }

  return true;
}

REGISTER_CUDA_OPERATOR(PowSum,
                       PowSumOp<float, CUDAContext>);
} // namespace caffe2
