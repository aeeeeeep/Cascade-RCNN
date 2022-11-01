# Cascade RCNN in mmdetection

## Environment

* python==3.8.10
* torch==1.10.0
* cuda==11.3
* mmcv==1.6.2
* mmcv-full==1.6.2
* mmdet==2.25.3

## Finsh
* 数据增强
* Soft-NMS
* CIoULoss(pass)
* half anchor
* Label Smoothing
* ConvNext Small
* IoU thr + 0.05
* GradientCumulativeOptimizerHook

## TODO
* 多尺度训练
* OHEM
* 可变形卷积
* 冻结backbone
