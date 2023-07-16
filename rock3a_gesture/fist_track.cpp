#include <math.h>
#include "fist_track.h"
#include "yolo-fastestv2-anchorfree.h"
#include "keypointsdetector.h"
#include "benchmark.h"
#include "cpu.h"
#include "widget.h"
#include "string"
extern Widget *w;

//const QString FistTrack::track_kind_Qstr[6] = {
//    "antiCW_circle", "CW_circle", "up_line", "down_line", "left_line", "right_line"
//};

const int FistTrack::draw_color[2][3] = {
    {0, 255, 0}, // green
    {255, 0, 0}  // blue
};

const char* FistTrack::class_names[2] = {
        "hand", "fist"
};

inline float distance_euclidean(nc::NdArray<float> point1, nc::NdArray<float> point2)
{
    return nc::sqrt(nc::sum(nc::square(point2 - point1)))(0, 0);
}

FistTrack::FistTrack()
{
    /* base param init that will be uesd in gesture kind */
    this->hand_count = 0;
    this->fist_count = 0;
    this->hand_max_count = 10;
    this->fist_max_count = 10;
    this->record_fist = false;
    get_screen_resolution();
}

void FistTrack::get_screen_resolution()
{
    QScreen *screen = QGuiApplication::primaryScreen();
    QRect rect = screen->availableGeometry() ;
    screen_width = rect.width();
    screen_height = rect.height();
    printf("[INFO] Current screen resolution: width=%d, height=%d\n", screen_width, screen_height);
}

void FistTrack::start(std::vector<int> num_threads,
                      int camera_id,
                      bool show_result,
                      VirtualDevice::DEVICE_TYPE virtual_device_type,
                      bool enable_kalman_filter,
                      float mouse_speed,
                      QString video_path)
{
    /* must be set at the begin
     * otherwsie if user double clicked the checkBox_start_stop, two signals will be sent
     * And the running_flag state: false(default)->false(second clicked: UnChecked)->running_flag=true
     * the start() won't be stopped.
     * Meanwhile the start() program will always remain one program in the slot's queue to wait. */
    this->running_flag = true;

    /* NCNN option 两个网络公用一个allocator*/
    this->g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    this->g_workspace_pool_allocator.set_size_compare_ratio(0.5f);
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads[0];
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_vulkan_compute = false;
    opt.use_bf16_storage = true;  
    opt.use_int8_inference = false;  
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = true;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;
    ncnn::set_cpu_powersave(0);     // all cores enabled
    ncnn::set_omp_dynamic(0);
//    ncnn::set_omp_num_threads(num_threads);
    this->g_blob_pool_allocator.clear();
    this->g_workspace_pool_allocator.clear();

    /* net init */
    yoloFastestv2AnchorFree detection_net(opt);             // hand detector net 
    KeypointsDetector keypoints_net(num_keypoints, num_threads[1], opt);    // keypoints net

    if (detection_net.loadModel("YOLO-FastestV2-AnchorFree-opt.param", "YOLO-FastestV2-AnchorFree-opt.bin") < 0 ||
        keypoints_net.load_model("LiteHRNetSimCC_w24-128x128.ncnn.param", "LiteHRNetSimCC_w24-128x128.ncnn.bin"))
    {
        emit signal_start_failed(-1);   // send failed signal
        return;
    }
    printf("[INFO] Ready to open the camera...\n");

    /* select intput: camera or video file */
    cv::VideoCapture cap;   // opencv video object
    if (camera_id == -1)
        cap.open(video_path.toStdString());
    else
        cap.open(camera_id);
    if (!cap.isOpened()){
        printf("[ERROR] Could not open video!\n");
        emit signal_start_failed(-2);
        return;
    }

    /* read input's size */
    int video_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int video_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    this->height = video_height;
    this->width = video_width;
    printf("[INFO] video width:%d, height:%d\n", video_width, video_height);

    /* create VirtualDevice object */
    if (virtual_device_type > 0) {
        try {
            virtual_input_device = new VirtualDevice(virtual_device_type);
        }  catch (const char* msg) {
            emit signal_start_failed(-3);
            return;
        }

    }

    /* start read input video and network interface,
     * then analyse track */
    cv::Mat frame;

    std::vector<TargetBox> boxes;
    std::vector<std::vector<cv::Point>> keypoints;
    std::vector<std::vector<float>> keypoints_scores;
    std::vector<std::vector<float>> figer_bend_value;
    GESTURE_KIND gesture_kind = NO_GESTURE;
    int show_dynamic_gesture_cnt = 0;  // frame time to show dynamic gesture result
#if SHOW_TIME_CONSUMED
    double while_start;
    double while_end = ncnn::get_current_time();
#endif
    goto_while_start:
    while(cap.read(frame))
    {        

        // {...} Mutex influence part
        {
            QMutexLocker locker(&this->lock);
            if (!this->running_flag) {
                printf("[INFO] Break the while loop...\n");
                break;
            }

        }
        /* horizontal filp the image */
        cv::flip(frame, frame, 1);
        /* interface */
#if SHOW_TIME_CONSUMED
        while_start = ncnn::get_current_time();
        printf("[INFO] ***cap read=%4.2fms  ", while_start-while_end);
        double start = ncnn::get_current_time();
#endif

        bool go_detect = false;
        if (keypoints_scores.size() > 0) {
            for (auto iter: keypoints_scores) {
                // float kpt_score_min = 1.0f;
                // float kpt_score_max = 0.0f;
                // float kpt_score_avg = 0.0f;
                for (auto score_: iter) {
                    // if (score_ > kpt_score_max)
                    //     kpt_score_max = score_;
                    // if (score_ < kpt_score_min)
                    //     kpt_score_min = score_;
                    // kpt_score_avg += score_;
                    if (score_ < keypoints_threshold) {
                        go_detect = true;   // if scores is small, then we enable hand detect
                        break;
                    }  
                }
            }
        }
        else
            go_detect = true;   // first has target
        if (go_detect) {
            printf("[INFO] Run detection to get real boxes!\n");
            detection_net.detection(frame, boxes, 0.7);
        }

        keypoints_net.get_keypoints(frame, boxes, keypoints, keypoints_scores);
        
        // 防止在手部快速移动时，当前帧也会跳过手部检测，仍然使用了上次预计的边界框，但是边界框中实际没有手
        // 此时关键点仍然会进行检测，导致关键点识别错乱，所以直接重新识别
        if (keypoints_scores.size() > 0) {
            // printf("[DEBUG] :");
            // float debug_min = 1.0f;
            // float debug_max = 0.0f;
            // float debug_avg = 0.0f;
            for (auto iter: keypoints_scores) {
                for (auto score_: iter) {
                    if (score_ < keypoints_threshold) {
                        goto goto_while_start;
                    }
                    // if (score_ > debug_max)
                    //     debug_max = score_;
                    // if (score_ < debug_min)
                    //     debug_min = score_;
                    // debug_avg += score_;
                }
            }
            // printf("\n--------max: %2.5f, min: %2.5f, avg: %2.5f\n", debug_max, debug_min, debug_avg / num_keypoints);
        }
        
        keypoints_net.get_roi_from_points(boxes, keypoints, 1.3);
        keypoints_net.get_fingers_bend(keypoints, keypoints_scores, figer_bend_value);

        gesture_kind = NO_GESTURE;
        get_gesture_kind_from_bend_value(figer_bend_value, gesture_kind);
        // take some time to show dynamic gesture result
        if (show_dynamic_gesture_cnt > 0)
            show_dynamic_gesture_cnt--;
        else
            emit signal_set_label_gesture_kind(gesture_str_map.at(gesture_kind));
        // printf("--------GESTURE_KIND: %d\n", gesture_kind);
#if SHOW_TIME_CONSUMED
        double end = ncnn::get_current_time();
        double forward_nms_time = end - start;
#endif
        /**
         * 1. get dynamic gesture
         * 2. send virtual keyboard value to kernel if enabled
         */
        if ((virtual_device_type == VirtualDevice::VIRTUAL_KEYBOARD ||
             virtual_device_type == VirtualDevice::VIRTUAL_MOUSE_KEYBOARD) &&
            boxes.size() == 1 )   // just for one hand
        {
            this->record_track(keypoints[0][0], gesture_kind);   // record landmark 0
            std::vector<float> dtw_dis;
            bool ret1 = false, ret2 = false;
            ret1 = this->get_dtw_dis(dtw_dis);
            ret2 = this->get_track_kind(gesture_kind);
            // if get gesture
            if (ret1 && ret2) {
                std::cout << "antiCW_circle: " << dtw_dis[0] << ", ";  // 逆时针
                std::cout << "CW_circle: " << dtw_dis[1] << ", ";      // 顺时针
                std::cout << "up_line: " << dtw_dis[2] << ", ";         // 上
                std::cout << "down_line: " << dtw_dis[3] << ", ";       // 下
                std::cout << "left_line: " << dtw_dis[4] << ", ";       // 左
                std::cout << "right_line: " << dtw_dis[5] << std::endl; // 右

                emit signal_set_label_gesture_kind(gesture_str_map.at(gesture_kind));
                show_dynamic_gesture_cnt = 50;
                // send virutal keyboard value
                virtual_input_device->emit_keyboard_event(gesture_key_map.at(gesture_kind));
            }
        }

#if SHOW_TIME_CONSUMED
        double boxes_start = ncnn::get_current_time();
#endif
        /**
         *  3. virutal mouse
         */
        if ((virtual_device_type == VirtualDevice::VIRTUAL_MOUSE ||
             virtual_device_type == VirtualDevice::VIRTUAL_MOUSE_KEYBOARD) &&
            boxes.size() == 1 )     // just for the one detection box
        {
            int is_click = 2;   // empty state
            if (gesture_kind == STATIC_GESTURE_8)
            {
                is_click = 1;   // click state
            }
            else if (gesture_kind == STATIC_GESTURE_7)
            {
                is_click = 0;   // move cursor state
            }
            // 1. Transfer the coordinate origin from the INDEX_FIGER_TIP keypoint(8),
            // and map a part of the center of the video to the desktop.
            // 2. Use the gesture to control curosr
            int cursor_x = (keypoints[0][8].x - this->width / 2) *
                    ((float)this->screen_width / (float)this->width / mouse_speed);
            int cursor_y = (keypoints[0][8].y - this->height / 2) *
                    ((float)this->screen_height / (float)this->height / mouse_speed);
            // current screen absolute x and y
            int new_cursor_x = cursor_x + this->screen_width / 2;
            int new_cursor_y = cursor_y + this->screen_height / 2;


            if (is_click == 0 || is_click == 1) {
//                virtual_input_device->emit_mouse_rel_event(new_cursor_x,
//                                                           new_cursor_y);
//                printf("[DEBUG] SetPos(%d, %d)\n", new_cursor_x, new_cursor_y);
                // Use Kalman filter, let cursor be stable
                if (enable_kalman_filter) {
                    // init the kalman filter if pointer is nullptr
                    if (enable_kalman_filter && kf == nullptr) {
                        cv::Point init_point(new_cursor_x, new_cursor_y);
                        kf = new KalmanFilter(init_point);
                        printf("[INFO] init the cursor's kalman filter\n");
                    }
                    // double kf_start = ncnn::get_current_time();
                    cv::Point kf_point(new_cursor_x, new_cursor_y);
                    kf->predict();
                    kf->update(kf_point);
                    kf->get_state(kf_point);
                    new_cursor_x = kf_point.x;
                    new_cursor_y = kf_point.y;
                    // double kf_end = ncnn::get_current_time();
                    // printf("[INFO] KF time:%4.2lfms\n", kf_end - kf_start);
                }
                // set absolute x, y
                QCursor::setPos(new_cursor_x, new_cursor_y);
                // mouse left button
                switch (is_click) {
                case 0:
                    virtual_input_device->emit_btn_left_event(0);     // release
                    break;
                case 1:
                    virtual_input_device->emit_btn_left_event(1);     // press
                    break;
                }
            }
        }

        /* draw detection result */
        for (int i = 0; i < boxes.size(); i++)
        {
            int box_cate = boxes[i].cate;

            /* draw detection result */
            if (show_result == true)
            {
                // std::cout<<boxes[i].x1<<" "<<boxes[i].y1<<" "<<boxes[i].x2<<" "<<boxes[i].y2
                //         <<" "<<boxes[i].score<<" "<<boxes[i].cate<<std::endl;

                char text[256] = {0};
                sprintf(text, "%s %.1f%%", class_names[box_cate], boxes[i].score * 100);

                int baseLine = 0;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                int x = boxes[i].x1;
                int y = boxes[i].y1 - label_size.height - baseLine;
                if (y < 0)
                    y = 0;
                if (x + label_size.width > frame.cols)
                    x = frame.cols - label_size.width;

                cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                              cv::Scalar(draw_color[box_cate][0], draw_color[box_cate][1], draw_color[box_cate][2]), -1); // 字体填充
                cv::putText(frame, text, cv::Point(x, y + label_size.height),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                cv::rectangle(frame, cv::Point(boxes[i].x1, boxes[i].y1),
                              cv::Point(boxes[i].x2, boxes[i].y2),
                              cv::Scalar(draw_color[box_cate][0], draw_color[box_cate][1], draw_color[box_cate][2]),
                              2, 2, 0); // 检测框
                // draw keypoints and skeleton
                keypoints_net.draw_keypoints(frame, keypoints, keypoints_scores, keypoints_threshold);
                if (gesture_kind == STATIC_GESTURE_1 || gesture_kind == STATIC_GESTURE_2)
                {
                    draw_menu(frame, keypoints, keypoints_scores, keypoints_threshold);
                }
            }
        }
#if SHOW_TIME_CONSUMED
        double boxes_end = ncnn::get_current_time();
#endif
        /* show result */
        if (show_result == true) {
//            cv::imshow("frame", frame);
//            if (cv::waitKey(1) == 27)
//                break;
            w->load_cv_image(frame);
        }
#if SHOW_TIME_CONSUMED
        while_end = ncnn::get_current_time();
        /* print time used */
        printf("[INFO] while=%4.2fms, forward+nms=%4.2fms, boxes=%4.2f, imshow=%4.2f\r\n",
                while_end-while_start, forward_nms_time, boxes_end-boxes_start, while_end-boxes_end);
#endif
    }
    printf("[INFO] while loop exited\n");
    /* after while() break, do some clean */
    cap.release();
    if (virtual_input_device != nullptr) {
        delete virtual_input_device;
        virtual_input_device = nullptr;
    }
    if (kf != nullptr) {
        delete kf;
        kf = nullptr;
    }
    printf("[INFO] Fist track terminated!\n");
    emit signal_start_exit();
}

/* just send break signal to while loop */
void FistTrack::terminate()
{
    QMutexLocker lokcer(&this->lock);
    this->running_flag = false;

    printf("[INFO] Terminate the fits track program......\n");
}

void FistTrack::record_track(cv::Point point, GESTURE_KIND gesture_kind)
{
    bool is_fist = false;
    if (gesture_kind == STATIC_GESTURE_10)
    {
        is_fist = true;
    }
    if (is_fist)
    {
        // 记录fist轨迹
        this->record_fist = true;
        this->fist_count++;
        this->recorded_track.emplace_back(point);
        // 记录fist时把hand_count清零，因为在记录fist过程中若出现hand误检，次数大于hand_max_count
        // 会导致fist轨迹判断提前结束
        this->hand_count = 0;
    }
    else if (is_fist == false && this->record_fist == true)
    {
        // 如果有fist检测框之后，后面出现hand检测框的时候
        // 对hand出现的次数进行计数，超出额定值则认为一段手势结束
        this->hand_count++;
    }

    if (this->hand_count > this->hand_max_count && this->fist_count > this->fist_max_count)
    {
        // 如果连续检测到多个hand，并且保存的fist检测框个数大于一定值(防止误检)则进行轨迹分析
        printf("[INFO] Fist Track End! Record fist count: %d\n", this->fist_count);
        // 分析保存的fist轨迹
        double analyse_start = ncnn::get_current_time();
        this->analyse_track();
        double analyse_end = ncnn::get_current_time();
        printf("[INFO]    Analyse time:%4.2fms\n", analyse_end-analyse_start);
        // 复位
        this->recorded_track.clear();
        this->record_fist = false;
        this->hand_count = 0;
        this->fist_count = 0;
    }
    else if (this->hand_count > this->hand_max_count && this->fist_count < this->fist_max_count)
    {
        // 若在一开始出现少数fist的误检，导致frecorded_track中存在误检框
        this->recorded_track.clear();
        this->record_fist = false;
        this->hand_count = 0;
        this->fist_count = 0;
    }
}

void FistTrack::analyse_track()
{
    /* 变为ndArray */
    nc::NdArray<float> main_track = nc::zeros<float>(this->recorded_track.size(), 2);
    for (int i = 0; i < this->recorded_track.size(); i++)
    {
        main_track(i, 0) = (float)this->recorded_track[i].x;
        main_track(i, 1) = (float)this->recorded_track[i].y;
    }
    /* 变为0-1区间内，保持高宽比不变 */
    this->rescale(main_track);
    /* 根据归一化后的检测框中心坐标xc yc，与预定义的轨迹计算DTW距离 */
    this->fist_track_dtw(main_track(main_track.rSlice(), {0, 2}));  // numpy=>main_track[:, 0:2]
}

void FistTrack::rescale(nc::NdArray<float> &input)
{
    float xmax = nc::max(input(input.rSlice(), 0))(0, 0);  // 返回NdArray再取值变为float
    float xmin = nc::min(input(input.rSlice(), 0))(0, 0);
    float ymax = nc::max(input(input.rSlice(), 1))(0, 0);
    float ymin = nc::min(input(input.rSlice(), 1))(0, 0);

    // 取差值最大的作为缩放因子
    float r = std::max(xmax - xmin, ymax - ymin);
    // tips: 切片只是拷贝，被赋值也不能改变原来变量的值，只能用put
    input.put(input.rSlice(), 0, (input(input.rSlice(), 0) - (xmax + xmin) / 2) / r);  // 将中心点平移到(0，0)处
    input.put(input.rSlice(), 1, (input(input.rSlice(), 1) - (ymax + ymin) / 2) / r);
}

void FistTrack::fist_track_dtw(nc::NdArray<float> input_xy)
{
    /* 获取起点 */
    float start_point_x = (input_xy(0, 0) + input_xy(1, 0)) / 2.0f;
    float start_point_y = (input_xy(0, 1) + input_xy(1, 1)) / 2.0f;
    int point_num = input_xy.shape().rows;
    printf("[INFO] Fist track length(DTW):%d\n", point_num);
    /* 1. 构造圆 */
    float circle_r = std::sqrt((start_point_x) * (start_point_x) +
                               (start_point_y) * (start_point_y));
    float angle = std::atan((start_point_y) / (start_point_x + 1e-10f));       // 防止除以0
    angle = angle * 180.f / 3.141592f;  // 弧度制变为角度制
    if (start_point_x < 0)
    {
        // 2、3象限+180°
        angle += 180.f;
    }
    std::map<GESTURE_KIND, nc::NdArray<float>> template_points;

    template_points[ANTI_CW] = this->get_circle_points(
                circle_r, point_num, angle, 360.0, 0.0, 0.0, false); // 逆时针
    template_points[CW] = this->get_circle_points(
        circle_r, point_num, angle, 360.0, 0.0, 0.0, true); // 顺时针

    /* 2. 构造直线 */
    template_points[UP_LINE] = this->get_xy_line_points(point_num, UP);
    template_points[DOWN_LINE] = this->get_xy_line_points(point_num, DOWN);
    template_points[LEFT_LINE] = this->get_xy_line_points(point_num, LEFT);
    template_points[RIGHT_LINE] = this->get_xy_line_points(point_num, RIGHT);
    // std::cout << input_xy<< std::endl;

    /* 3. 计算轨迹与模板的DTW距离 */
    float dis = 0.0f;
    float min_dis = 999999.9f;
    GESTURE_KIND min_dis_kind;
    std::vector<float> dtw_dis_;
    std::map<GESTURE_KIND, nc::NdArray<float>>::iterator iter;
    for (iter = template_points.begin(); iter != template_points.end(); iter++)
    {
        dis = this->compute_dtw(iter->second,
                                input_xy,
                                distance_euclidean);
        // emplace_back可以减少析构函数调用
        dtw_dis_.emplace_back(dis); // 保存DTW距离
        if (dis < min_dis)
        {
            min_dis = dis;
            min_dis_kind = iter->first;
        }
    }
    this->dtw_dis.emplace(dtw_dis_);
    this->track_kind.emplace(min_dis_kind); // 保存轨迹匹配的结果
}


nc::NdArray<float> FistTrack::get_circle_point(float r, float angle,
                                               float x0,
                                               float y0)
{
    float angle_pi = angle / 180.f * 3.141592f; // 角度制变为弧度制
    float x = x0 + r * std::cos(angle_pi);
    float y = y0 + r * std::sin(angle_pi);
    nc::NdArray<float> point = {{x, y}};
    return point;
}

nc::NdArray<float> FistTrack::get_circle_points(float r, int num,
                                                float start_angle,
                                                float angle_range,
                                                float x0,
                                                float y0,
                                                bool clock_wise)
{
    nc::NdArray<float> points = nc::zeros<float>(num, 2);
    float per_angle = angle_range / (float)num;
    float angle = start_angle;
    for (int i = 0; i < num; i++)
    {
        points.put(i, points.cSlice(), this->get_circle_point(r, angle, x0, y0));
        if (clock_wise)
            angle += per_angle;
        else
            angle -= per_angle;
    }
    return points;
}

nc::NdArray<float> FistTrack::get_xy_line_points(int num, LineDirect direct)
{
    float step = 1.0f / (float)num;
    nc::NdArray<float> points = nc::zeros<float>(num, 2);
    float x = 0.0;
    float y = 0.0;
    float step_x = 0;
    float step_y = 0;
    switch (direct)
    {
    case UP:
        // (0,0.5)->(0,-0.5)
        x = 0.0;
        y = 0.5;
        step_x = 0;
        step_y = -step;
        break;
    case DOWN:
        // (0,-0.5)->(0,0.5)
        x = 0.0;
        y = -0.5;
        step_x = 0;
        step_y = step;
        break;
    case LEFT:
        // (0.5,0)->(-0.5,0)
        x = 0.5;
        y = 0.0;
        step_x = -step;
        step_y = 0;
        break;
    case RIGHT:
        // (-0.5,0)->(0.5,0)
        x = -0.5;
        y = 0.0;
        step_x = step;
        step_y = 0;
        break;
    default:
        printf("[ERROR] Invalid direct!\n");
        break;
    }
    for (int i = 0; i < num; i++)
    {
        points(i, 0) = x;
        points(i, 1) = y;
        x += step_x;
        y += step_y;
    }
    return points;
}

float FistTrack::intersection_area(const TargetBox &a, const TargetBox &b)
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

float FistTrack::compute_iou(TargetBox bbox1, TargetBox bbox2)
{
    // 交集
    float inter_area = intersection_area(bbox1, bbox2);
    // 并集
    float union_area = bbox1.area() + bbox2.area() - inter_area;
    float IoU = inter_area / union_area;

    return IoU;
}

void FistTrack::iou_filter(const std::vector<TargetBox> &bboxes,
                           std::vector<TargetBox> &main_group,
                           std::vector<TargetBox> &other_group,
                           float iou_thread)
{
    int last_main = 0;
    main_group.emplace_back(bboxes[last_main]);
    for (int i = 0; i < bboxes.size() - 1; i++)
    {
        // 当前bbox和下一个bbox计算iou
        float iou = this->compute_iou(bboxes[last_main], bboxes[i + 1]);
        // 如果iou大于阈值，则分为一组
        if (iou > iou_thread)
        {
            last_main = i + 1;
            main_group.emplace_back(bboxes[i + 1]);
        }
        else
            other_group.emplace_back(bboxes[i + 1]);
    }
}

std::vector<TargetBox> FistTrack::iou_filters(std::vector<TargetBox> &bboxes)
{
    std::vector<std::vector<TargetBox>> groups;

    std::vector<TargetBox> group1;
    std::vector<TargetBox> group2;
    std::vector<TargetBox> group_temp;
    // 先分一次组 保存组1
    this->iou_filter(bboxes, group1, group2);
    groups.emplace_back(group1);
    // 为了防止主要轨迹被分为到group2中，所以继续分组
    while (group2.size() > this->fist_max_count)
    {
        group_temp = group2;
        // 先清空再重新给iou_filter存放
        group1.clear();
        group2.clear();
        // 不断对组2进行分组 直到分完为止
        this->iou_filter(group_temp, group1, group2);
        groups.emplace_back(group1);
    }
    // 找到最长的组作为轨迹组
    int max_len = 0;
    int index = 0;
    for (int i = 0; i < groups.size(); i++)
    {
        int l = groups[i].size();
        if (l > max_len)
        {
            max_len = l;
            index = i;
        }
    }
    return groups[index];
}

// TODO 优化
float FistTrack::compute_dtw(nc::NdArray<float> &A, nc::NdArray<float> &B,
                             float (*dis_func)(nc::NdArray<float> point1, nc::NdArray<float> point2))
{
    int N_A = A.shape().rows;
    int N_B = B.shape().rows;
    int i = 0;
    int j = 0;

    nc::NdArray<float> D = nc::zeros<float>(N_A, N_B); // 累计距离矩阵

    D(0, 0) = dis_func(A.row(0), B.row(0));
    // 最左边一列
    for (i = 1; i < N_A; i++)
    {
        D(i, 0) = D(i - 1, 0) + dis_func(A.row(i), B.row(0));
    }
    // 下面一行
    for (j = 1; j < N_B; j++)
    {
        D(0, j) = D(0, j - 1) + dis_func(A.row(0), B.row(j));
    }
    // 中间
    for (i = 1; i < N_A; i++)
        for (j = 1; j < N_B; j++)
        {
            // TODO 这里找最小值时把最小值路径记录下来，后面回溯的时候就不用再找了
            D(i, j) = dis_func(A.row(i), B.row(j)) + std::min({D(i, j - 1), D(i - 1, j - 1), D(i - 1, j)});
        }

    /* 路径回溯 */
    int count = 0; // 统计多少个配对点
    // 起始位置
    i = N_A - 1;
    j = N_B - 1;
    nc::NdArray<float> d = nc::zeros<float>(1, std::max(N_A, N_B) * 2); // 记录匹配距离
    std::vector<int> path_row;                                          // 记录匹配点，暂时用不上所以没输出
    std::vector<int> path_col;
    while (true)
    {
        if (i > 0 && j > 0) // 走到中间区域
        {
            // 保存当前路径
            path_row.emplace_back(i);
            path_col.emplace_back(j);
            // 寻找下一路径
            float m = std::min({D(i, j - 1), D(i - 1, j - 1), D(i - 1, j)});
            if (m == D(i, j - 1)) // 最小值在左边
            {
                d(0, count) = D(i, j) - D(i, j - 1);
                j = j - 1;
                count++;
            }
            else if (m == D(i - 1, j - 1)) // 左下
            {
                d(0, count) = D(i, j) - D(i - 1, j - 1);
                i = i - 1;
                j = j - 1;
                count++;
            }
            else if (m == D(i - 1, j)) // 下
            {
                d(0, count) = D(i, j) - D(i - 1, j);
                i = i - 1;
                count++;
            }
        }
        else if (i == 0 && j == 0) // 走到头
        {
            path_row.emplace_back(i);
            path_col.emplace_back(j);
            d(0, count) = D(i, j);
            count++;
            break;
        }
        else if (i == 0) // 走到最下面
        {
            path_row.emplace_back(i);
            path_col.emplace_back(j);
            d(0, count) = D(i, j) - D(i, j - 1);
            j = j - 1;
            count++;
        }
        else if (j == 0) // 走到最左边
        {
            path_row.emplace_back(i);
            path_col.emplace_back(j);
            d(0, count) = D(i, j) - D(i - 1, j);
            i = i - 1;
            count++;
        }
    }
    float mean = nc::sum(d)(0, 0) / (float)count;
    return mean;
}

void FistTrack::get_gesture_kind_from_bend_value(std::vector<std::vector<float>> figer_bend_value,
                                                 GESTURE_KIND &gesture_kind)
{
    // only one hand rightnow
    if (figer_bend_value.size() != 1)
        return;

    if (figer_bend_value[0][0] > 0.6f &&    // thumb NOT bend
        figer_bend_value[0][1] > 0.0f &&    // index NOT bend
        figer_bend_value[0][2] > 0.0f &&    // middle NOT bend
        figer_bend_value[0][3] > 0.0f &&    // ring NOT bend
        figer_bend_value[0][4] > 0.0f       // pinky NOT bend
        )
    {
        gesture_kind = STATIC_GESTURE_1;
    }
    else if (figer_bend_value[0][0] < 0.6f &&    // thumb bend
             figer_bend_value[0][1] > 0.0f &&    // index NOT bend
             figer_bend_value[0][2] > 0.0f &&    // middle NOT bend
             figer_bend_value[0][3] > 0.0f &&    // ring NOT bend
             figer_bend_value[0][4] > 0.0f       // pinky NOT bend
            )
    {
        gesture_kind = STATIC_GESTURE_2;
    }
    else if (figer_bend_value[0][0] < 0.6f &&    // thumb bend
             figer_bend_value[0][1] < 0.0f &&    // index bend
             figer_bend_value[0][2] > 0.0f &&    // middle NOT bend
             figer_bend_value[0][3] > 0.0f &&    // ring NOT bend
             figer_bend_value[0][4] > 0.0f       // pinky NOT bend
            )
    {
        gesture_kind = STATIC_GESTURE_3;
    }
    else if (figer_bend_value[0][0] > 0.6f &&    // thumb NOT bend
             figer_bend_value[0][1] > 0.0f &&    // index NOT bend
             figer_bend_value[0][2] > 0.0f &&    // middle NOT bend
             figer_bend_value[0][3] < 0.0f &&    // ring bend
             figer_bend_value[0][4] < 0.0f       // pinky bend
            )
    {
        gesture_kind = STATIC_GESTURE_4;
    }
    else if (figer_bend_value[0][0] < 0.6f &&    // thumb bend
             figer_bend_value[0][1] > 0.0f &&    // index NOT bend
             figer_bend_value[0][2] > 0.0f &&    // middle NOT bend
             figer_bend_value[0][3] < 0.0f &&    // ring bend
             figer_bend_value[0][4] < 0.0f       // pinky bend
            )
    {
        gesture_kind = STATIC_GESTURE_5;
    }
    else if (figer_bend_value[0][0] > 0.6f &&    // thumb NOT bend
             figer_bend_value[0][1] < 0.0f &&    // index bend
             figer_bend_value[0][2] < 0.0f &&    // middle bend
             figer_bend_value[0][3] < 0.0f &&    // ring bend
             figer_bend_value[0][4] > 0.0f       // pinky NOT bend
            )
    {
        gesture_kind = STATIC_GESTURE_6;
    }
    else if (figer_bend_value[0][0] > 0.6f &&    // thumb NOT bend
             figer_bend_value[0][1] > 0.0f &&    // index NOT bend
             figer_bend_value[0][2] < 0.0f &&    // middle bend
             figer_bend_value[0][3] < 0.0f &&    // ring bend
             figer_bend_value[0][4] < 0.0f       // pinky bend
            )
    {
        gesture_kind = STATIC_GESTURE_7;
    }
    else if (figer_bend_value[0][0] < 0.6f &&    // thumb bend
             figer_bend_value[0][1] > 0.0f &&    // index NOT bend
             figer_bend_value[0][2] < 0.0f &&    // middle bend
             figer_bend_value[0][3] < 0.0f &&    // ring bend
             figer_bend_value[0][4] < 0.0f       // pinky bend
            )
    {
        gesture_kind = STATIC_GESTURE_8;
    }
    else if (figer_bend_value[0][0] > 0.6f &&    // thumb NOT bend
             figer_bend_value[0][1] < 0.8f &&    // index bend
             figer_bend_value[0][2] < 0.0f &&    // middle bend
             figer_bend_value[0][3] < 0.0f &&    // ring bend
             figer_bend_value[0][4] < 0.0f       // pinky bend
             )
    {
        gesture_kind = STATIC_GESTURE_9;
    }
    else if (figer_bend_value[0][0] < 0.6f &&    // thumb bend
             figer_bend_value[0][1] < 0.0f &&    // index bend
             figer_bend_value[0][2] < 0.0f &&    // middle bend
             figer_bend_value[0][3] < 0.0f &&    // ring bend
             figer_bend_value[0][4] < 0.0f       // pinky bend
            )
    {
        gesture_kind = STATIC_GESTURE_10;
    }

}

void FistTrack::draw_menu(cv::Mat &frame, std::vector<std::vector<cv::Point>> &points,
                          std::vector<std::vector<float>> &scores,
                          float threshold)
{
    int menu_num = 5;
    int index_finger[2] = {0, 8};   // index finger
    int theta = 20;                 // degree
    double radian_theta = theta * 3.14f / 180;
    int fontFace = cv::FONT_HERSHEY_COMPLEX;
    double fontScale = 0.5;
    int SIZE = 15;
    int menu_circle_size[menu_num];
    std::string menu_name[menu_num];
    cv::Point menu_pos[menu_num];
    // init the menu size and name
    for (int i = 0; i < menu_num; i++) {
        menu_circle_size[i] = SIZE;
        menu_name[i] = "menu" + std::to_string(i+1);
    }

    // only one hand
    if (points.size() != 1 || scores[0][index_finger[0]] < threshold || scores[0][index_finger[1]] < threshold)
        return;

    cv::Point p1 = points[0][index_finger[0]];
    cv::Point p2 = points[0][index_finger[1]];
    int mid = menu_num / 2;
    int length = 1.3f * std::sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));   // 超出向量长度
    menu_pos[mid] = cv:: Point(p1.x, p1.y - length);       // vertical menu pos

    // calculate other menu pos，从中间往两边计算menu_pos
    for (int i = 1; i <= mid; i++) {
        int temp_x = length * std::sin(i * radian_theta);
        int temp_y = length * std::cos(i * radian_theta);
        // left
        menu_pos[mid - i] = cv::Point(menu_pos[mid].x - temp_x, menu_pos[mid].y + (length - temp_y));
        // right
        if ((mid + i) < menu_num) // avoid overwrite the mid when menu_num is even number
            menu_pos[mid + i] = cv::Point(menu_pos[mid].x + temp_x, menu_pos[mid].y + (length - temp_y));
    }

    int v0_x = p2.x - p1.x;
    int v0_y = p2.y - p1.y;
    int v1_x = menu_pos[0].x - p1.x;
    int v1_y = menu_pos[0].y - p1.y;
    // the angle between menu[0] ane finger
    float cos_theta_finger = (float)(v0_x * v1_x + v0_y * v1_y) /   \
                             (std::sqrt(v0_x * v0_x + v0_y * v0_y) * std::sqrt(v1_x * v1_x + v1_y * v1_y) + 1e-3);


    int finger_index = (std::acos(cos_theta_finger) + radian_theta / 2.0f) / radian_theta;
    // 判断食指的向量是在第0个菜单向量的左边还是右边，如果是左边，那么变为负数，因为左边finger_index也会得到2， 3
    if (v0_x * v1_y - v0_y * v1_x > 0)  // > 0 表示v0在 v1的左边
        finger_index = -finger_index;
    if (finger_index >= 0 && finger_index < menu_num)
        menu_circle_size[finger_index] *= 2;

    for (int i = 0; i < menu_num; i++) {
        cv::circle(frame, menu_pos[i], menu_circle_size[i], cv::Scalar(51, 156, 255), -1);
        // 测量文字的矩形区域的大小
        int baseline;
        cv::Size textSize = cv::getTextSize(menu_name[i], fontFace, fontScale, 1, &baseline);
        // 计算文字的原点位置，使其居中于圆内
        cv::Point origin;
        origin.x = menu_pos[i].x - textSize.width / 2;
        origin.y = menu_pos[i].y + textSize.height / 2;
        cv::putText(frame, menu_name[i], origin, fontFace, fontScale, cv::Scalar(255, 255, 255), 1);
    }

}
