#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QPushButton>
#include <QRadioButton>
#include <QCheckBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QLabel>
#include <QSpinBox>
#include <QThread>
#include <QCloseEvent>
#include <QComboBox>

#include "fist_track.h"


#define ENABLE_VIRTUAL_CHOOSE 0
#define ENABLE_CHOOSE_FILE 0

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

// to avoid fist_track.h and widget.h include each other
class FistTrack;

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

    void load_cv_image(cv::Mat &cv_image);

private:
    Ui::Widget *ui;

    QWidget *widget_form_layout;
    QWidget *widget_grid_layout;
    QGridLayout *grid_layout;
    QGridLayout *grid_layout_form;
    QFormLayout *form_layout;
    QSpinBox *spinBox_box_num_threads;             // detection net threads
    QSpinBox *spinBox_kpt_num_threads;             // keypoints net threads
    QSpinBox *spinBox_camera_id;                // camera id
    QDoubleSpinBox *doubleSpinBox_mouse_speed;              // mouse speed

    QCheckBox *checkBox_enable_kalman_filter;   // enable kalman filter
    QCheckBox *checkBox_show_result;            // is show result
    QCheckBox *checkBox_start_stop;             // start/stop network interface
#if ENABLE_VIRTUAL_CHOOSE
    QComboBox *combobox_select_virtual_device;  // select virtual device: none, mouse, keyboard
#endif

#if ENABLE_CHOOSE_FILE
    QPushButton *button_open_video_file;        // open video file button
    QLabel *label_current_file;                 // to show selected video file
#endif
    QLabel *label_result_image;                 // to show result image
    QLabel *label_gesture_kind;                 // to show gesture kind
    QThread fist_track_thread;
    FistTrack *fist_track;

    std::vector<int> num_threads = {2, 3};      // hand detection net thread, hand keypoints thread
    int camera_id = 0;
    bool show_result = true;
    VirtualDevice::DEVICE_TYPE virtual_device_type = VirtualDevice::VIRTUAL_NONE;
    bool enable_kalman_filter = true;
    float mouse_speed = 0.6f;
    QString video_path = "";

signals:
    void start_fist_track(std::vector<int> num_threads,
                          int camera_id,
                          bool show_result,
                          VirtualDevice::DEVICE_TYPE virtual_device_type,
                          bool enable_kalman_filter,
                          float mouse_speed,
                          QString video_path);


private slots:
    void spinBox_box_num_threads_valueChanged(int);
    void spinBox_kpt_num_threads_valueChanged(int);
    void spinBox_camera_id_valueChanged(int);
    void checkBox_show_result_stateChanged(int);
    void checkBox_start_stop_stateChanged(int);
    void slot_start_failed(int err_num);
    void slot_start_exit();
    void slot_set_label_gesture_kind(QString);
#if ENABLE_VIRTUAL_CHOOSE
    void slot_combobox_select_virtual_device(int);
#endif
#if ENABLE_CHOOSE_FILE
    void slot_button_open_video_file();
#endif
    void slot_doubleSpinBox_mouse_speed(double speed_ratio);
    void slot_checkBox_enable_kalman_filter(int state);

protected:
    void closeEvent(QCloseEvent *event);

};
#endif // WIDGET_H
