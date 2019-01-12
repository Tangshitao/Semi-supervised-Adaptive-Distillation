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
#include "box_annotator_ohem_op.h"

namespace caffe2 {



template<>
bool BoxAnnotatorOHEMOp<float, CUDAContext>::RunOnDevice() {

  auto& rois = Input(0);
  auto& roi_per_loss = Input(1);
  auto& weights = Input(2);
  auto* label_weights = Output(0);
  auto* bbox_weights = Output(1);
  
  int N,bbox_channels_;
  N = rois.dim32(0);
  bbox_channels_=weights.dim32(1);

  label_weights->Resize(N);
  bbox_weights->ResizeLike(weights);

  Tensor<CPUContext> bottom_rois_cpu(rois),bottom_loss_cpu(roi_per_loss),weights_cpu(weights);
  const float* bottom_rois=bottom_rois_cpu.data<float>();
  const float* bottom_loss=bottom_loss_cpu.data<float>();
  const float* weights_data=weights_cpu.data<float>();
  
  int num_imgs = -1;
  for (int n = 0; n < N; n++) {
    num_imgs = bottom_rois[n*5] > num_imgs ? bottom_rois[n*5] : num_imgs;
  }
  num_imgs++;

  vector<int> sorted_idx(N);
  for (int i = 0; i < N; i++) {
    sorted_idx[i] = i;
  }
  std::sort(sorted_idx.begin(), sorted_idx.end(),
    [bottom_loss](int i1, int i2) {
      return bottom_loss[i1] > bottom_loss[i2];
  });

  Tensor<CPUContext> bbox_weights_cpu,label_weights_cpu;
  label_weights_cpu.Resize(N);
  bbox_weights_cpu.ResizeLike(weights);
  auto* bbox_weights_cpu_data=bbox_weights_cpu.mutable_data<float>();
  auto* label_weights_cpu_data=label_weights_cpu.mutable_data<float>();

  for(int i=0;i<N;i++){
    label_weights_cpu_data[i]=0;
    for(int j=0;j<bbox_channels_;j++){
      bbox_weights_cpu_data[i*bbox_channels_+j]=0;
    }
  }

  vector<int> number_left(num_imgs, roi_per_img_);
  for (int i = 0; i < N; i++) {
    int index = sorted_idx[i];
    int batch_ind = bottom_rois[index*5];
    if (number_left[batch_ind] > 0) {
      number_left[batch_ind]--;
      label_weights_cpu_data[index]=1;
      for (int j = 0; j < bbox_channels_; j++) {
        int bbox_index = index*bbox_channels_+j;
          bbox_weights_cpu_data[bbox_index] =weights_data[bbox_index];
          //CAFFE_ENFORCE(false,weights_data[bbox_index]);
      }
    }
  }

  label_weights->CopyFrom(label_weights_cpu);
  bbox_weights->CopyFrom(bbox_weights_cpu);

  return true;
}


REGISTER_CUDA_OPERATOR(BoxAnnotatorOHEM,
                       BoxAnnotatorOHEMOp<float, CUDAContext>);

}  // namespace caffe2
