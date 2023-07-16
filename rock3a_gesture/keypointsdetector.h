#ifndef KEYPOINTSDETECTOR_H
#define KEYPOINTSDETECTOR_H

#include <iostream>
#include <QObject>
#include <vector>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include "yolo-fastestv2-anchorfree.h"
#include "net.h"
#include "benchmark.h"
#include "cpu.h"


/**
 *  KeypointsDetector
 */
class KeypointsDetector: public QObject
{
    Q_OBJECT
private:
    int net_input_height = 128;
    int net_input_width = 128;
    float box_scale = 1.1f;

    TargetBox adjust_box(TargetBox &box_in);

public:
    int num_keypoints;
    int num_outputs;
    ncnn::Net net;
//    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
//    ncnn::PoolAllocator g_workspace_pool_allocator;

    KeypointsDetector(int nkpts, int num_threads, ncnn::Option opt);
    ~KeypointsDetector();

    int load_model(const char* paramPath, const char* binPath);

    /**
     * get all boxes keypoionts
     */
    void get_keypoints(cv::Mat image_o, std::vector<TargetBox> boxes,
                       std::vector<std::vector<cv::Point>> &points,
                       std::vector<std::vector<float>> &scores);

    void postHandle(const ncnn::Mat *out, float simcc_split_ratio, cv::Mat &trans,
                    std::vector<cv::Point> &points, std::vector<float> &scores);

    /* get whether five fingers are bent */
    void get_fingers_bend(std::vector<std::vector<cv::Point>> &points,
                          std::vector<std::vector<float>> &scores,
                          std::vector<std::vector<float>> &bend_value);

    /* draw keypoints and skeleton */
    void draw_keypoints(cv::Mat &image_o,
                        std::vector<std::vector<cv::Point>> &points,
                        std::vector<std::vector<float>> &scores,
                        float threshold);

    /* rect roi area by keypoints */
    void get_roi_from_points(std::vector<TargetBox> &boxes,
                             std::vector<std::vector<cv::Point>> &points,
                             float ratio);

};

#endif // KEYPOINTSDETECTOR_H
