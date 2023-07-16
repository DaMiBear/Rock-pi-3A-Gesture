#ifndef __YOLOFASTESTV2ANCHORFREE_H
#define __YOLOFASTESTV2ANCHORFREE_H

#include "net.h"
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

#define SHOW_TIME_CONSUMED 0

class TargetBox
{
private:


public:
    int x1;
    int y1;
    int x2;
    int y2;

    int cate;
    float score;
    TargetBox(): x1(0), y1(0), x2(0), y2(0) {};
    TargetBox(int x1, int y1, int x2, int y2): x1(x1), y1(y1), x2(x2), y2(y2) {};
    int area() { return getWidth() * getHeight(); };
    int getWidth() { return (x2 - x1); };
    int getHeight() { return (y2 - y1); };

    float getCenterX() { return (x1 + x2) / 2.0f; };
    float getCenterY() { return (y1 + y2) / 2.0f; };
    int getLeftX() { return x1; };
    int getTopY() { return y1; };
    int getRightX() { return x2; };
    int getBottomY() { return y2; };
};

class yoloFastestv2AnchorFree
{
private:
    const char *inputName;
    const char *outputName1;
    const char *outputName2;

    int numAnchor;
    int numOutput;
    int numCategory;
    int inputWidth, inputHeight;

    float nmsThresh;

    int nmsHandle(std::vector<TargetBox> &tmpBoxes, std::vector<TargetBox> &dstBoxes);
    int getCategory(const float *values, int index, int &category, float &score);
    int predHandle(const ncnn::Mat *out, std::vector<TargetBox> &dstBoxes,
                   const float scaleW, const float scaleH, const float thresh);

public:
    ncnn::Net net;

    yoloFastestv2AnchorFree(ncnn::Option opt);
    ~yoloFastestv2AnchorFree();

    int loadModel(const char* paramPath, const char* binPath);
    int detection(const cv::Mat& srcImg, std::vector<TargetBox> &dstBoxes,
                        const float thresh = 0.3);
};

#endif // __YOLOFASTESTV2ANCHORFREE_H
