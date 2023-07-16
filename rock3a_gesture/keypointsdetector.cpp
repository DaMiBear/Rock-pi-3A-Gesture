#include "keypointsdetector.h"
#include "NumCpp.hpp"
#include "fist_track.h"
/**
 *  KeypointsDetector
 */
KeypointsDetector::KeypointsDetector(int nkpts, int num_threads, ncnn::Option opt)
{
    num_outputs = 2;    // 输出节点数 x和y
    num_keypoints = nkpts;

    opt.num_threads = num_threads;
    net.opt = opt;
}

KeypointsDetector::~KeypointsDetector()
{
    printf("\n[INFO] Destroy KeypointsDetector ...\n");
}

int KeypointsDetector::load_model(const char* paramPath, const char* binPath)
{
    // LiteHRNetSimCC_w24-128x128.ncnn.param
    // LiteHRNetSimCC_w24-128x128.ncnn.bin
    if (net.load_param(paramPath) < 0)
        return -1;
    if (net.load_model(binPath) < 0)
        return -2;
    printf("[INFO] load KeypointsDetector success!\r\n");
    return 0;
}

TargetBox KeypointsDetector::adjust_box(TargetBox &box_in)
{

    int xmin = box_in.getLeftX();
    int ymin = box_in.getTopY();
    int xmax = box_in.getRightX();
    int ymax = box_in.getBottomY();
    int h = box_in.getHeight();
    int w = box_in.getWidth();
    float hw_ratio = (float)net_input_height / (float)net_input_width;

    if ((float)h / (float)w > hw_ratio)
    {
        // 需要在w方向padding
        float wi = (float)h / hw_ratio;
        float pad_w = (wi - w) / 2.0f;
        xmin = xmin - pad_w;
        xmax = xmax + pad_w;
    }
    else
    {
        // 需要在h方向padding
        float hi = (float)w * hw_ratio;
        float pad_h = (hi - h) / 2.0f;
        ymin = ymin - pad_h;
        ymax = ymax + pad_h;
    }
    TargetBox box_out(xmin, ymin, xmax, ymax);
    return box_out;
}

void getMax(const float *in_arr, int arr_len, float &max_val, int &max_index)
{
    float max = 0;
    int max_i = 0;
    for (int i = 0; i < arr_len; i++) {
        if (in_arr[i] > max) {
            max = in_arr[i];
            max_i = i;
        }
    }
    max_val = max;
    max_index = max_i;
}

/**
 * @brief 后处理LiteSimCC网络的输出（网络最后一层为softmax）
 * 
 * @param out ncnn网络节点的输出数组
 * @param simcc_split_ratio simcc ratio 128x128 = 2.0
 * @param trans 预处理时的逆仿射变换矩阵
 * @param points 保存关键点的向量
 * @param scores 保存关键点概率的向量，不算是置信度
 */
void KeypointsDetector::postHandle(const ncnn::Mat *out,
                                   float simcc_split_ratio,
                                   cv::Mat &trans,
                                   std::vector<cv::Point> &points,
                                   std::vector<float> &scores)
{
    // out[i] shape: [d: 1, c: 1, h: nkpt, w: net_input_w(h) * ratio]
    int net_num_kpts = out[0].h;
    if (net_num_kpts != num_keypoints) {
        printf("[ERROR] net's num_keypoints != num_keypoints");
        return;
    }
    cv::Mat kpts_mat = cv::Mat::ones(cv::Size(net_num_kpts, 3), CV_32F);
    for (int kpt_id = 0; kpt_id < net_num_kpts; kpt_id++)
    {   
        int values_len_x = out[0].w;
        int values_len_y = out[1].w;

        const float *values_x = out[0].row(kpt_id);   // return shape [net_input_w(h) * ratio]
        const float *values_y = out[1].row(kpt_id);   // return shape [net_input_w(h) * ratio]
        float max_val_x = 0;
        float max_val_y = 0;
        int max_index_x = 0;
        int max_index_y = 0;
        float score = 0;
        getMax(values_x, values_len_x, max_val_x, max_index_x);
        getMax(values_y, values_len_y, max_val_y, max_index_y);

        float pred_x = max_index_x / simcc_split_ratio;
        float pred_y = max_index_y / simcc_split_ratio;
        score = std::max(max_val_x, max_val_y);
        scores.emplace_back(score);

        // 把关键点放射变换回原图位置  trans shape[2, 3]
        double new_x = trans.at<double>(0, 0) * pred_x + \
                      trans.at<double>(0, 1) * pred_y + \
                      trans.at<double>(0, 2) * 1.0f;
        double new_y = trans.at<double>(1, 0) * pred_x + \
                      trans.at<double>(1, 1) * pred_y + \
                      trans.at<double>(1, 2) * 1.0f;
        points.emplace_back(cv::Point(new_x, new_y));
    }

}

void KeypointsDetector::get_keypoints(cv::Mat image_o,
                                      std::vector<TargetBox> boxes,
                                      std::vector<std::vector<cv::Point>> &points,
                                      std::vector<std::vector<float>> &scores)
{
    points.clear();
    scores.clear();
    // 对每一个box进行关键点检测
    for (int i = 0; i < boxes.size(); i++)
    {
        /* 图片预处理 */
        // box调整为网络输入大小，高宽比不变
        TargetBox adjusted_box = adjust_box(boxes[i]);
        int src_xmin = adjusted_box.getLeftX();
        int src_ymin = adjusted_box.getTopY();
        int src_xmax = adjusted_box.getRightX();
        int src_ymax = adjusted_box.getBottomY();
        float src_xcenter = adjusted_box.getCenterX();
        float src_ycenter = adjusted_box.getCenterY();
        // 计算放射变换矩阵
        int src_w = src_xmax - src_xmin;
        int src_h = src_ymax - src_ymin;
        if (box_scale != 1.0f)
        {
            src_w = (float)src_w * box_scale;
            src_h = (float)src_h * box_scale;
        }
        cv::Point2f src[3];
        src[0] = cv::Point2f(src_xcenter, src_ycenter);                // center
        src[1] = cv::Point2f(src_xcenter, src_ycenter - src_h / 2.0f); // top middle
        src[2] = cv::Point2f(src_xcenter + src_w / 2.0f, src_ycenter); // right middle

        cv::Point2f dst[3];
        dst[0] = cv::Point2f((float)(net_input_width - 1) / 2.0f, (float)(net_input_height - 1) / 2.0f); // center
        dst[1] = cv::Point2f((float)(net_input_width - 1) / 2.0f, 0.0f);                                 // top middle
        dst[2] = cv::Point2f(net_input_width - 1, (float)(net_input_height - 1) / 2.0f);                 // right middle
        cv::Mat trans = cv::getAffineTransform(src, dst);                                                // 计算正向仿射变换矩阵
        cv::Mat reverse_trans = cv::getAffineTransform(dst, src);                                        // 计算逆向仿射变换矩阵，方便后续还原
//        std::cout << "reverse_trans = (numpy)" << std::endl << cv::format(reverse_trans, cv::Formatter::FMT_NUMPY) << std::endl << std::endl;
        // 进行放射变换
        cv::Mat resize_img;
        cv::warpAffine(image_o, resize_img, trans, cv::Size(net_input_width, net_input_height), cv::INTER_LINEAR);

        /* forward */
        ncnn::Mat net_img_input = ncnn::Mat::from_pixels(resize_img.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                         net_input_width, net_input_height);
        const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
        const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
        net_img_input.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Mat out[2];
        ncnn::Extractor ex = net.create_extractor(); // 几乎不占用时间

        ex.input("in0", net_img_input);
#if SHOW_TIME_CONSUMED
        double forward_start = ncnn::get_current_time();
#endif
        ex.extract("out0", out[0]);
        ex.extract("out1", out[1]);
#if SHOW_TIME_CONSUMED
        double forward_end = ncnn::get_current_time();
        printf("[INFO] keypoints forward = %4.2fms\n", forward_end - forward_start);
#endif
        std::vector<cv::Point> points_tmp;
        std::vector<float> scores_tmp;
        postHandle(out, 2.0, reverse_trans, points_tmp, scores_tmp);
        points.emplace_back(points_tmp);
        scores.emplace_back(scores_tmp);
    }
}

void KeypointsDetector::get_fingers_bend(std::vector<std::vector<cv::Point>> &points,
                                         std::vector<std::vector<float>> &scores,
                                         std::vector<std::vector<float>> &bend_value)
{
    const int fingers_index [][2] = {
        {3, 4}, {0, 2},         // thumb finger
        {7, 8}, {0, 6},         // index finger
        {11, 12}, {0, 10},      // middle finger
        {15, 16}, {0, 14},      // ring finger
        {19, 20}, {0,18}        // pinky finger
    };
    bend_value.clear();
    for (int s = 0; s < points.size(); s++) {
        std::vector<float> per_hand_bend_value;
//        printf("[DEBUG] cos theta:");
        for(int i = 0; i <= 4; i++) {
            int v0_x = points[s][fingers_index[i * 2][1]].x - points[s][fingers_index[i * 2][0]].x;
            int v0_y = points[s][fingers_index[i * 2][1]].y - points[s][fingers_index[i * 2][0]].y;

            int v1_x = points[s][fingers_index[i * 2 + 1][1]].x - points[s][fingers_index[i * 2 + 1][0]].x;
            int v1_y = points[s][fingers_index[i * 2 + 1][1]].y - points[s][fingers_index[i * 2 + 1][0]].y;
            // 1e-2 in case divide zero and the length of vector is small enough
            float cos_theta = (float)(v0_x * v1_x + v0_y * v1_y) /   \
                              (std::sqrt(v0_x * v0_x + v0_y * v0_y) * std::sqrt(v1_x * v1_x + v1_y * v1_y) + 1e-2);
//            printf("[DEBUG]  %d = %4.2f", i, cos_theta);
            per_hand_bend_value.emplace_back(cos_theta);
        }
        bend_value.emplace_back(per_hand_bend_value);
//        printf("\r\n");
    }



}

void KeypointsDetector::draw_keypoints(cv::Mat &image_o,
                                       std::vector<std::vector<cv::Point>> &points,
                                       std::vector<std::vector<float>> &scores,
                                       float threshold)
{
    static const int sk_len = 21;
    static cv::Scalar color[] = {cv::Scalar(0, 153, 255), cv::Scalar(0, 0, 255), cv::Scalar(255, 102, 153),
                                 cv::Scalar(255, 153, 0), cv::Scalar(102, 204, 0), cv::Scalar(0, 255, 255),
                                 cv::Scalar(255, 255, 204)};
    static int sk[][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4},
                          {0, 5}, {5, 6}, {6, 7}, {7, 8},
                          {9, 10}, {10, 11}, {11, 12},
                          {13, 14}, {14, 15}, {15, 16},
                          {0, 17}, {17, 18}, {18, 19}, {19, 20},
                          {5, 9}, {9, 13}, {13, 17}};
    static int sk_color[] = {0, 1, 1, 1,
                             0, 2, 2, 2,
                             3, 3, 3,
                             4, 4, 4,
                             0, 5, 5, 5,
                             0, 0, 0};

    for (int h = 0; h < points.size(); h++) {
        // points
        for (int k = 0; k < num_keypoints; k++) {
            if (scores[h][k] > threshold) {
                cv::circle(image_o, points[h][k], 2, cv::Scalar(0, 255, 0), 8);
            }
        }
        // skeleton
        for (int k = 0; k < sk_len; k++) {
            float score1 = scores[h][sk[k][0]];
            float score2 = scores[h][sk[k][1]];
            if (score1 > threshold && score2 > threshold) {
                cv::line(image_o, points[h][sk[k][0]], points[h][sk[k][1]], color[sk_color[k]], 3);
            }
        }
    }

}

void KeypointsDetector::get_roi_from_points(std::vector<TargetBox> &boxes,
                                            std::vector<std::vector<cv::Point>> &points,
                                            float ratio)
{
    int boxes_num = boxes.size();
    for (int i = 0 ; i < boxes_num; i++) {
        int ps[num_keypoints];
        int x_max = 0;
        int y_max = 0;
        int x_min = 999999;
        int y_min = 999999;
        for (auto iter: points[i]) {
            if (iter.x > x_max)
                x_max = iter.x;
            if (iter.x < x_min)
                x_min = iter.x;
            if (iter.y > y_max)
                y_max = iter.y;
            if (iter.y < y_min)
                y_min = iter.y;
        }
        int w = x_max - x_min;
        int h = y_max - y_min;
        int new_w = (float)w * ratio;
        int new_h = (float)h * ratio;
        int pad_w = (new_w - w) / 2;
        int pad_h = (new_h - h) / 2;
        x_min = x_min - pad_w;
        x_max = x_max + pad_w;
        y_min = y_min - pad_h;
        y_max = y_max + pad_h;
        boxes[i].x1 = x_min;
        boxes[i].y1 = y_min;
        boxes[i].x2 = x_max;
        boxes[i].y2 = y_max;
    }
}
