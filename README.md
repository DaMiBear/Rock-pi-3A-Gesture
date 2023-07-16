一个简单的手部检测网络和手部关键点检测，以及aarch64 Linux的手势demo Qt代码。

事先说明，代码中涉及的所有内容都是本人第一次接触，有可能存在不恰当的地方。分享出来仅仅作为学习使用。

## 包括四部分：

- `YOLO-FastestV2-AnchorFree`一个目标检测网络
- `Lite-HRNet-SimCC`一个关键点检测网络
- `rock3a_gesture`在Rock pi 3A上跑的Qt程序
- `net_benchmark`在Rock pi 3A上跑的网络推理耗时程序
- `generate_hand_keypoints_datasets`产生一个简单的数据集