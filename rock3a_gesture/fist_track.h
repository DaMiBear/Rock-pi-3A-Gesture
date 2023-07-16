#ifndef FIST_TRACK_H
#define FIST_TRACK_H

#include "yolo-fastestv2-anchorfree.h"
#include "linux_virtual_device.h"
#include "kalman_filter.h"
#include "NumCpp.hpp"
#include <algorithm>
#include <queue>

#include <QMutex>
#include <QMutexLocker>
#include <QObject>
#include <QImage>
#include <QPixmap>
#include <QLabel>
#include <QString>
#include <QScreen>
#include <QGuiApplication>
#include "widget.h"


inline float distance_euclidean(nc::NdArray<float> point1, nc::NdArray<float> point2);

enum LineDirect
{
    UP = 0,
    DOWN,
    LEFT,
    RIGHT,
};

class Point
{
public:
    float x;
    float y;
    Point(float x, float y)
    {
        this->x = x;
        this->y = y;
    }
};

class FistTrack : public QObject
{
    Q_OBJECT
public:
    enum GESTURE_KIND{
        // dynamic gesture
        ANTI_CW,
        CW,
        UP_LINE,
        DOWN_LINE,
        LEFT_LINE,
        RIGHT_LINE,
        // static gesture
        STATIC_GESTURE_1,
        STATIC_GESTURE_2,
        STATIC_GESTURE_3,
        STATIC_GESTURE_4,
        STATIC_GESTURE_5,
        STATIC_GESTURE_6,
        STATIC_GESTURE_7,
        STATIC_GESTURE_8,
        STATIC_GESTURE_9,
        STATIC_GESTURE_10,
        // Nothing
        NO_GESTURE,
    };

    FistTrack();

    /* stop fist track while program */
    void terminate();

    /**
     * @brief 记录每一帧的point，并判断fist手势轨迹结束
     *
     * @param point
     */
    void record_track(cv::Point point, GESTURE_KIND gesture_kind);

    // inline需要在这里定义
    inline bool get_dtw_dis(std::vector<float> &out)
    {
        if (this->dtw_dis.empty())
            return false;
        else
        {
            out = this->dtw_dis.front();
            this->dtw_dis.pop();
            return true;
        }
    }

    inline bool get_track_kind(GESTURE_KIND &out)
    {
        if (this->track_kind.empty()) {
            return false;
        }
        else {
            out = this->track_kind.front();
            this->track_kind.pop();
            return true;
        }
    }

public slots:
    void start(std::vector<int> num_threads,
               int camera_id,
               bool show_result,
               VirtualDevice::DEVICE_TYPE virtual_device_type,
               bool enable_kalman_filter,
               float mouse_speed,
               QString video_path);
signals:
    void signal_start_failed(int err_num);
    void signal_start_exit();
    void signal_set_label_gesture_kind(QString);

private:
    int height;                         // 原图的高
    int width;                          // 原图的宽
    int screen_height;                  // current screen height (maybe smaller)
    int screen_width;                   // current screen width (maybe smaller)
    int hand_count;                     // hand检测计数
    int fist_count;                     // fist检测计数
    int hand_max_count;                 // hand检测阈值
    int fist_max_count;                 // fist检测阈值
    bool record_fist;                   // true表示正在进行fist轨迹记录
    bool running_flag;                  // control while break in start()
    int num_keypoints = 21;
    float keypoints_threshold = 0.03;
    KalmanFilter *kf = nullptr;                   // Kalman Filter pointer

    /* uinput virtual device */
    VirtualDevice *virtual_input_device = nullptr;

    QMutex lock;                        // Mutex
    std::vector<cv::Point> recorded_track; // 存放轨迹
    std::queue<std::vector<float>> dtw_dis;         // 存放fist轨迹与预定义轨迹的DTW距离
    std::queue<GESTURE_KIND> track_kind;        // 存放轨迹种类

    // map: gesture kind <---> virtual keyboard value
    const std::map<GESTURE_KIND, VirtualDevice::KEYBOARD_KEY_TYPE> gesture_key_map = {
        {ANTI_CW, VirtualDevice::VIRTUAL_KEY_ESC},
        {CW, VirtualDevice::VIRTUAL_KEY_ENTER},
        {UP_LINE, VirtualDevice::VIRTUAL_KEY_UP},
        {DOWN_LINE, VirtualDevice::VIRTUAL_KEY_DOWN},
        {LEFT_LINE, VirtualDevice::VIRTUAL_KEY_LEFT},
        {RIGHT_LINE, VirtualDevice::VIRTUAL_KEY_RIGHT},
    };
    // map: gesture kind <---> Qt string
    const std::map<GESTURE_KIND, QString> gesture_str_map = {
        {ANTI_CW, "Anti-clockwise"},
        {CW, "Clockwise"},
        {UP_LINE, "Up"},
        {DOWN_LINE, "Down"},
        {LEFT_LINE, "Left"},
        {RIGHT_LINE, "Right"},
        {STATIC_GESTURE_1, "Static Gesture 1"},
        {STATIC_GESTURE_2, "Static Gesture 2"},
        {STATIC_GESTURE_3, "Static Gesture 3"},
        {STATIC_GESTURE_4, "Static Gesture 4"},
        {STATIC_GESTURE_5, "Static Gesture 5"},
        {STATIC_GESTURE_6, "Static Gesture 6"},
        {STATIC_GESTURE_7, "Static Gesture 7"},
        {STATIC_GESTURE_8, "Static Gesture 8"},
        {STATIC_GESTURE_9, "Static Gesture 9"},
        {STATIC_GESTURE_10, "Static Gesture 10"},
        {NO_GESTURE, "No Gesture"}
    };
    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;

    std::queue<cv::Mat> result_queue;   // 储存检测结果cv::Mat队列
//    static const QString track_kind_Qstr[6];
    static const char* class_names[2];   // classes
    static const int draw_color[2][3];   // target color

    /* get the current screen's resolution */
    void get_screen_resolution();

    /**
     * @brief 对获得的轨迹进行分析。过滤、DTW
     *
     */
    void analyse_track();

    /**
     * @brief 从半径为r的圆中取一个点
        以左上角为坐标原点
        x------->
        y
        |
        |
        v
     *
     * @param r 圆的半径
     * @param angel 极坐标角度 顺时针与x轴夹角
     * @param x0 圆心横坐标
     * @param y0 圆心纵坐标
     * @return nc::NdArray<float> 设定的xy坐标点
     */
    nc::NdArray<float> get_circle_point(float r, float angle,
                                        float x0 = 0.0,
                                        float y0 = 0.0);

    /**
     * @brief 从一个圆中均匀取Num个点 从start_angle角度开始
        以左上角为坐标原点
        x------->
        y
        |
        |
        v
     *
     * @param r 圆的半径
     * @param num 要获得点的个数
     * @param start_angle 要从哪个角度开始取点 与x轴顺时针方向的夹角
     * @param angle_range 取点的角度范围 与start_angle顺时针方向的夹角
     * @param x0 圆心x
     * @param y0 圆心y
     * @param clock_wise True顺时针 False逆时针
     * @return nc::NdArray<float> 圆上的点集
     */
    nc::NdArray<float> get_circle_points(float r, int num,
                                         float start_angle = 0.0,
                                         float angle_range = 360.0,
                                         float x0 = 0.0,
                                         float y0 = 0.0,
                                         bool clock_wise = false);

    /**
     * @brief 在[0-1]范围内，x轴左右方向，或y轴上下方向取num个点
        以左上角为坐标原点
        x------->
        y
        |
        |
        v
     *
     * @param num 取点个数
     * @param direct 取点方向 UP DOWN LEFT RIGHT
     * @return nc::NdArray<float> 点xy坐标
     */
    nc::NdArray<float> get_xy_line_points(int num, LineDirect direct);

    /**
     * @brief 计算两个框的交集
     *
     * @param a
     * @param b
     * @return float
     */
    float intersection_area(const TargetBox &a, const TargetBox &b);
    /**
     * @brief 计算两个框的iou
     *
     * @param box1
     * @param box2
     * @return float iou
     */
    float compute_iou(TargetBox bbox1, TargetBox bbox2);

    /**
     * @brief 根据连续的两两框之间的iou是否有交集 分成若干组，返回最长的一组
     *
     * @param bboxes
     * @return std::vector<TargetBox> iou有交集的连续框中最长的一组
     */
    std::vector<TargetBox> iou_filters(std::vector<TargetBox> &bboxes);

    /**
     * @brief 根据连续的两两之间框的iou是否有交集，分成两组
     *
     * @param bboxes [in]
     * @param main_group [out]组1
     * @param other_group [out]组2
     * @param iou_thread 交集阈值
     */
    void iou_filter(const std::vector<TargetBox> &bboxes,
                    std::vector<TargetBox> &main_group,
                    std::vector<TargetBox> &other_group,
                    float iou_thread = 0.1);

    /**
     * @brief 计算fist轨迹与预定义轨迹之间的DTW距离
     *
     * @param input_xy fist中心坐标轨迹
     */
    void fist_track_dtw(nc::NdArray<float> input_xy);

    float compute_dtw(nc::NdArray<float> &A, nc::NdArray<float> &B,
                      float (*dis_func)(nc::NdArray<float> point1, nc::NdArray<float> point2));

    /**
     * @brief 进行归一化，但保持高宽比不变，轨迹中心移动到原点(0,0)处
     *
     * @param input
     */
    void rescale(nc::NdArray<float> &input);

    /**
     * @brief get static gesture kind from the fingre bend value
     *
     * @param std::vector<std::vector<float>>: bend value each finger each hand
     * @param GESTURE_KIND: return kind
     */
    void get_gesture_kind_from_bend_value(std::vector<std::vector<float>> figer_bend_value,
                                          GESTURE_KIND &gesture_kind);
    /**
     * @brief draw menu on the image
     */
    void draw_menu(cv::Mat &frame, std::vector<std::vector<cv::Point>> &points,
                   std::vector<std::vector<float>> &scores,
                   float threshold);
};

#endif
