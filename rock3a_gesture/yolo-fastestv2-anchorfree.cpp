#include <math.h>
#include <algorithm>
#include "yolo-fastestv2-anchorfree.h"
#include "benchmark.h"
#include <cassert>
#include <iostream>


//模型的参数配置
yoloFastestv2AnchorFree::yoloFastestv2AnchorFree(ncnn::Option opt)
{
    printf("[INFO] Creat yoloFastestv2AnchorFree Detector...\n");
    //输出节点数
    numOutput = 2;
    //anchor num
    numAnchor = 1;
    //类别数目
    numCategory = 1;
    //NMS阈值
    nmsThresh = 0.3;

    //模型输入尺寸大小
    inputWidth = 352;
    inputHeight = 352;

    //模型输入输出节点名称
    inputName = (const char*)"input.1";
    outputName1 = (const char*)"789"; //14x22
    outputName2 = (const char*)"790"; //7x11

    //打印初始化相关信息
    printf("[INFO] inputWidth:%d inputHeight:%d\n", inputWidth, inputHeight);
    net.opt = opt;
}

yoloFastestv2AnchorFree::~yoloFastestv2AnchorFree()
{
    printf("[INFO] Destroy yoloFastestv2AnchorFree Detector...\n");
}

//ncnn 模型加载
int yoloFastestv2AnchorFree::loadModel(const char* paramPath, const char* binPath)
{
    printf("[INFO] Ncnn mode init:\n%s\n%s\n", paramPath, binPath);

    if (net.load_param(paramPath) < 0)
        return -1;
    if (net.load_model(binPath) < 0)
        return -2;

    printf("[INFO] Ncnn model init sucess...\n");
    return 0;
}

float intersection_area(const TargetBox &a, const TargetBox &b)
{
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b)
{
    return (a.score > b.score);
}

//NMS处理
int yoloFastestv2AnchorFree::nmsHandle(std::vector<TargetBox> &tmpBoxes,
                             std::vector<TargetBox> &dstBoxes)
{
    std::vector<int> picked;

    sort(tmpBoxes.begin(), tmpBoxes.end(), scoreSort);

    for (int i = 0; i < tmpBoxes.size(); i++) {
        int keep = 1;
        for (int j = 0; j < picked.size(); j++) {
            //交集
            float inter_area = intersection_area(tmpBoxes[i], tmpBoxes[picked[j]]);
            //并集
            float union_area = tmpBoxes[i].area() + tmpBoxes[picked[j]].area() - inter_area;
            float IoU = inter_area / union_area;

            if(IoU > nmsThresh && tmpBoxes[i].cate == tmpBoxes[picked[j]].cate) {
                keep = 0;
                break;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }

    for (int i = 0; i < picked.size(); i++) {
        dstBoxes.push_back(tmpBoxes[picked[i]]);
    }

    return 0;
}

//检测类别分数处理
int yoloFastestv2AnchorFree::getCategory(const float *values, int index, int &category, float &score)
{
    float tmp = 0;
    float objScore  = values[4 * numAnchor + index];  // 第index个锚框的模板的obj
    for (int i = 0; i < numCategory; i++) {  // 遍历所有的类别概率，找出最大的
        float clsScore = values[4 * numAnchor + numAnchor + i];
        clsScore *= objScore;

        if(clsScore > tmp) {
            score = clsScore;
            category = i;

            tmp = clsScore;
        }
    }

    return 0;
}

void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z=0; z<m.d; z++)
        {
            for (int y=0; y<m.h; y++)
            {
                for (int x=0; x<m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
}

//特征图后处理
int yoloFastestv2AnchorFree::predHandle(const ncnn::Mat *out, std::vector<TargetBox> &dstBoxes,
                              const float scaleW, const float scaleH, const float thresh)
{    //do result
    for (int i = 0; i < numOutput; i++) {
        int stride;
        int outW, outH, outC;
        // 在导出ONNX模型的时候，模型输出的形状是(bs,14,22,4+1+2)和(bs,7,11,4+1+2)
        outH = out[i].c;
        outW = out[i].h;
        outC = out[i].w;

        assert(inputHeight / outH == inputWidth / outW);
        stride = inputHeight / outH;

        for (int h = 0; h < outH; h++) {
            // out[i].channel(h)这一行原理：
            // 先创建一个Mat，因为定义了模板和类型转换函数，所以返回的是Mat中data的地址
            const float* values = out[i].channel(h);  // channel return: shape(22,4+1+2) or (11,4+1+2) 但先指向第0行
            for (int w = 0; w < outW; w++) {
                for (int b = 0; b < numAnchor; b++) {
                    //float objScore = values[4 * numAnchor + b];
                    TargetBox tmpBox;
                    int category = -1;
                    float score = -1;

                    getCategory(values, b, category, score);

                    if (score > thresh) {
                        float bcx, bcy, bw, bh;

                        bcx = (values[b * 4 + 0] + w) * stride;
                        bcy = (values[b * 4 + 1] + h) * stride;
                        bw = exp(values[b * 4 + 2]) * stride;
                        bh = exp(values[b * 4 + 3]) * stride;
                        // 从网络输入大小变为原图大小
                        tmpBox.x1 = (bcx - 0.5 * bw) * scaleW;
                        tmpBox.y1 = (bcy - 0.5 * bh) * scaleH;
                        tmpBox.x2 = (bcx + 0.5 * bw) * scaleW;
                        tmpBox.y2 = (bcy + 0.5 * bh) * scaleH;
                        tmpBox.score = score;
                        tmpBox.cate = category;

                        dstBoxes.push_back(tmpBox);
                    }
                }
                values += outC;  // 再指向下一行
            }
        }
    }
    return 0;
}

int yoloFastestv2AnchorFree::detection(const cv::Mat& srcImg, std::vector<TargetBox> &dstBoxes, const float thresh)
{
    dstBoxes.clear();

    float scaleW = (float)srcImg.cols / (float)inputWidth;
    float scaleH = (float)srcImg.rows / (float)inputHeight;

    //resize of input image data
    ncnn::Mat inputImg = ncnn::Mat::from_pixels_resize(srcImg.data, ncnn::Mat::PIXEL_BGR2RGB,\
                                                       srcImg.cols, srcImg.rows, inputWidth, inputHeight);

    //Normalization of input image data
//    const float mean_vals[3] = {0.f, 0.f, 0.f};
//    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
    inputImg.substract_mean_normalize(mean_vals, norm_vals);  // 0-255 -> 0-1

    ncnn::Mat out[2];
    ncnn::Extractor ex = net.create_extractor();

    //set input tensor
    ex.input(inputName, inputImg);
#if SHOW_TIME_CONSUMED
    double forward_start = ncnn::get_current_time();
#endif
    //forward
    ex.extract(outputName1, out[0]); //14x22
    ex.extract(outputName2, out[1]); //7x11
#if SHOW_TIME_CONSUMED
    double forward_end = ncnn::get_current_time();
#endif
    std::vector<TargetBox> tmpBoxes;
    //特征图后处理
#if SHOW_TIME_CONSUMED
    double predhandle_start = ncnn::get_current_time();
#endif
    predHandle(out, tmpBoxes, scaleW, scaleH, thresh);
    //NMS
    nmsHandle(tmpBoxes, dstBoxes);
#if SHOW_TIME_CONSUMED
    double predhandle_end = ncnn::get_current_time();
    printf("[INFO] forward=%4.2fms, predhandle=%4.2fms\r\n", forward_end-forward_start, predhandle_end-predhandle_start);
#endif
    return 0;
}
