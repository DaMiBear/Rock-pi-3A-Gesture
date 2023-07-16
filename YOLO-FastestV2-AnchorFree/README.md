# YOLO-FastestV2-AnchorFree

这个代码参考了[YOLO-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2)、[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)、[YOLOV3-SPP](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/yolov3_spp)。非常感谢这些代码作者🌼。

主要把YOLO-FastestV2中的Anchor-Based改为了YOLOX中的Anchor-Free，加入了一些数据增强。我自己是用来检测手部的，自己建立的数据集，网络很小，能做一些简单的检测，置信度阈值可能需要拉高一些，避免误报。

- `train.py`文件只支持一个GPU，在训练前需要准备自己的数据集，YOLO格式或者COCO格式的数据集仓库中都有对应的读取代码（默认是COCO格式，YOLO格式需要在路径的地方进行一些小修改）。数据集的路径等信息要在`data/mydata.data`中进行修改。`cfg/hyp.yaml`中是一些参数的设置。训练的数据使用了[wandb](https://wandb.ai/)记录，不想用的可以注释掉相关的代码。
- `draw_some_datasets.py`可以绘制一些经过数据增强后的图片。
- `get_mAP.py`就是对数据集进行一遍mAP的计算。
- `predict_video.py`可以读取视频，然后利用网络对视频中的目标进行检测，保存结果到`out`路径中。
- `pytorch2onnx.py`可以将网络导出为ONNX格式，然后利用`toncnn`中的工具进一步将网络模型导出为NCNN格式用于部署到其他设备中。



