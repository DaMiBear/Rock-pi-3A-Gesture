# Qt Hand Gesture

**这个项目很多第三方库需要自己交叉编译为库文件，并有对应的CMake文件后，才能通过CMake正确编译。如OpenCV、NumCpp、NCNN、FFmepg。**

这个项目的功能就是读取检测网络和关键点网络的NCNN文件，用于手势识别，实现一些简单的静态手势和手部轨迹的分类，进而对应不同的简单控制。创建了额外的一个线程用于网络推理。
