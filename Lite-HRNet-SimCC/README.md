# Lite-HRNet-SimCC

## 该项目主要参考以下仓库
* [HRNet](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_keypoint/HRNet)
* [Lite-HRNet](https://github.com/HRNet/Lite-HRNet)
* [SimCC](https://github.com/leeyegy/SimCC)

## 相关说明

- `train.py`：用于单卡训练，需要修改自己的数据集路径（*训练时建议使用大数据集上的预训练模型，也可以是其他任务的。或者先去掉SimCC部分进行热力图方式的训练，然后再将权重用于初始化Lite-HRNet-SimCC网络中除SimCC以外的部分*）。
- `validation.py`：用于验证mAP。
- `predict.py`：用于读取一张图片进行关键点绘制，仅限图像主题为主要目标。
- `save_some_train_images.py`：用于保存一些数据增强后的图片。
- `toTorchScript.py`：用于将网络导出为PNNX格式，详见https://github.com/pnnx/pnnx
