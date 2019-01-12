# Semi-supervised Adaptive Distillation

Semi-supervised Adaptive Distillation is a model compression method for object detection. Please refer to our [paper]() for more details. The code is implemented with official [detectron](https://github.com/facebookresearch/Detectron) and [Caffe2](https://github.com/caffe2/caffe2).


## Main results
| Student model | Baseline mAP | Teacher model | Baseline mAP | Student mAP after distillation |
|---------------|--------------|---------------|--------------|--------------------------------|
| ResNet-50     | 34.3         | ResNet-101    | 36.0         | 36.5                           |
| ResNet-101    | 34.4         | ResNext-101   | 36.6         | 36.8                           |

We use the input scale of 600 for ResNet-50 and 500 for ResNet-101. The results are reported on COCO mini-val.

## Requirements
We include the custom caffe2 in our code. The requirements is the same as the offical detectron and Caffe2. To run our codes with official Caffe2, please add 2 operators. One is located at `caffe2/modules/detectron/pow_sum_op.h` and the other is located at `caffe2/modules/detectron/sigmoid_focal_distillation_loss_op.h`.

## Installation
Please follow the official installation step of [detectron](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md).

## Resources
1. Teacher model: ResNet-101. [BaiduYun](), [Google Drive]()
2. Teacher model: ResNext-101. [BaiduYun](), [Google Drive]()
3. The annotation file for COCO 2017 unlabel data produced by the ResNet-101 teacher model above. [BaiduYun](), [Google Drive]()
4. The annotation file for COCO 2017 unlabel data produced by the ResNext-101 teacher model above. [BaiduYun](), [Google Drive]()
5. Student model after distillation: ResNet-50. [BaiduYun](). [Google Drive]()
5. Student model after distillation: ResNet-101. [BaiduYun](). [Google Drive]()

## Training
```
python2 tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/focal_distillation/retinanet_R-50-FPN_distillation.yaml \
    --teacher_cfg configs/focal_distillation/retinanet_R-101-FPN_1x_teacher.yaml
```
We assume the weight file for teacher model is located at `weights/R101_600/model_final.pkl` and the annotations file is located at `lib/datasets/data/annotations/image_info_unlabeled2017_r101_600.json`. 


