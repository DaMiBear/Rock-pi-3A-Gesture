#include "widget.h"
#include "./ui_widget.h"
#include <QFileDialog>
#include <QMetaType>
#include <QProxyStyle>
#include "fist_track.h"
#include "linux_virtual_device.h"

#define DEFAULT_IMAGE ":/images/snowman.png"

#define COMBOBOX_ITEM_NONE       "None"
#define COMBOBOX_ITEM_MOUSE      "Mouse"
#define COMBOBOX_ITEM_KEYBOARD   "Keyboard"
#define COMBOBOX_ITEM_MOUSE_KEYBOARD    "Mouse&Keyboard"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    /* set widget size */
    int ui_fixed_size_width = 1000;
    int ui_fixed_size_height = 650;
    this->setFixedSize(ui_fixed_size_width, ui_fixed_size_height);
    widget_grid_layout = new QWidget(this);
    widget_grid_layout->setGeometry(0, 0, ui_fixed_size_width, ui_fixed_size_height);
    /* set user input:num thread and camera id */
    widget_form_layout = new QWidget();
    spinBox_box_num_threads = new QSpinBox();
    spinBox_box_num_threads->setRange(1, 4);
    spinBox_box_num_threads->setSingleStep(1);
    spinBox_box_num_threads->setValue(2);
//    spinBox_box_num_threads->setFixedWidth(130);
    spinBox_kpt_num_threads = new QSpinBox();
    spinBox_kpt_num_threads->setRange(1, 4);
    spinBox_kpt_num_threads->setSingleStep(1);
    spinBox_kpt_num_threads->setValue(3);

    spinBox_camera_id = new QSpinBox();
    spinBox_camera_id->setRange(-1, 10);
    spinBox_camera_id->setSingleStep(1);
    spinBox_camera_id->setValue(0);
    // mouse speed, small -> fast
    doubleSpinBox_mouse_speed = new QDoubleSpinBox();
    doubleSpinBox_mouse_speed->setRange(0.1, 1.0);
    doubleSpinBox_mouse_speed->setSingleStep(0.1);
    doubleSpinBox_mouse_speed->setValue(0.6);
    doubleSpinBox_mouse_speed->setFixedWidth(150);
    doubleSpinBox_mouse_speed->setDecimals(1);  // 0.x
    // show result
    checkBox_show_result = new QCheckBox();
    checkBox_show_result->setChecked(show_result); // init state
    // enabel kalman filter
    checkBox_enable_kalman_filter = new QCheckBox();
    checkBox_enable_kalman_filter->setChecked(true);
    // choose virtual device
#if ENABLE_VIRTUAL_CHOOSE
    combobox_select_virtual_device = new QComboBox();
    combobox_select_virtual_device->addItem(COMBOBOX_ITEM_NONE);
    combobox_select_virtual_device->addItem(COMBOBOX_ITEM_MOUSE);
    combobox_select_virtual_device->addItem(COMBOBOX_ITEM_KEYBOARD);
    combobox_select_virtual_device->addItem(COMBOBOX_ITEM_MOUSE_KEYBOARD);
    combobox_select_virtual_device->setFixedWidth(150);
#endif
#if ENABLE_CHOOSE_FILE
    // open a video file
    button_open_video_file = new QPushButton();
    button_open_video_file->setText("Open File");
    // show current selected video file
    label_current_file = new QLabel();
    label_current_file->setFixedSize(100, 50);
    label_current_file->setWordWrap(true);  // auto \n
    label_current_file->setAlignment(Qt::AlignLeft);
#endif
    // add gird to replace form layout, grid can be align!
    grid_layout_form = new QGridLayout();
    grid_layout_form->addWidget(new QLabel("Show Result:"), 0, 0);
    grid_layout_form->addWidget(checkBox_show_result, 0, 1);

    grid_layout_form->addWidget(new QLabel("Kalman Filter:"), 1, 0);
    grid_layout_form->addWidget(checkBox_enable_kalman_filter, 1, 1);

    grid_layout_form->addWidget(new QLabel("Box Threads:"), 2, 0);
    grid_layout_form->addWidget(spinBox_box_num_threads, 2, 1);

    grid_layout_form->addWidget(new QLabel("Kpt Threads:"), 3, 0);
    grid_layout_form->addWidget(spinBox_kpt_num_threads, 3, 1);

    grid_layout_form->addWidget(new QLabel("Camera ID:"), 4, 0);
    grid_layout_form->addWidget(spinBox_camera_id, 4, 1);

    grid_layout_form->addWidget(new QLabel("Mouse Speed:"), 5, 0);
    grid_layout_form->addWidget(doubleSpinBox_mouse_speed, 5, 1);
#if ENABLE_VIRTUAL_CHOOSE
    grid_layout_form->addWidget(new QLabel("Virtual Device:"), 6, 0);
    grid_layout_form->addWidget(combobox_select_virtual_device, 6, 1);
#endif
#if ENABLE_CHOOSE_FILE
    grid_layout_form->addWidget(new QLabel("Select File:"), 7, 0);
    grid_layout_form->addWidget(button_open_video_file, 7, 1);
    grid_layout_form->addWidget(new QLabel("Current File:"), 8, 0);
    grid_layout_form->addWidget(label_current_file, 8, 1);
#endif
    widget_form_layout->setLayout(grid_layout_form);

    /* start or stop */
    checkBox_start_stop = new QCheckBox();
    // checkBox_start_stop->setGeometry(50, 500, 200, 50);
    checkBox_start_stop->setText("Start");
    checkBox_start_stop->setChecked(false);

    /* show detection result image */
    label_result_image = new QLabel();
    // label_result_image->setGeometry(350, 0, 500, 500);
    label_result_image->setScaledContents(false);   // set true will fill the label area not keep ratio
    QPixmap label_result_pixmap(DEFAULT_IMAGE);
    label_result_image->setPixmap(label_result_pixmap);

    /* show gesture result text */
    label_gesture_kind = new QLabel();
    label_gesture_kind->setText("Gesture: ");

    /* Grid layout set */
    grid_layout = new QGridLayout();
    grid_layout->addWidget(widget_form_layout, 0, 0);
    grid_layout->addWidget(checkBox_start_stop, 1, 0, Qt::AlignCenter);
    grid_layout->addWidget(label_result_image, 0, 1);
    grid_layout->addWidget(label_gesture_kind, 1, 1, Qt::AlignCenter);
    grid_layout->setRowStretch(0, 7);
    grid_layout->setRowStretch(1, 1);
    grid_layout->setColumnStretch(0, 1);
    grid_layout->setColumnStretch(1, 5);

    widget_grid_layout->setLayout(grid_layout);


    /* FistTrack thread */
    fist_track = new FistTrack();
    fist_track->moveToThread(&fist_track_thread);

    /* connect signals and slots */
    // destory object and object's thread
    connect(&fist_track_thread, SIGNAL(finished()),
            fist_track, SLOT(deleteLater()));
    connect(&fist_track_thread, SIGNAL(finished()),
            &fist_track_thread, SLOT(deleteLater()));
    // Note: registered user define type
    qRegisterMetaType<VirtualDevice::DEVICE_TYPE>("VirtualDevice::DEVICE_TYPE");
    qRegisterMetaType<std::vector<int>>("std::vector<int>");

    // thread start
    connect(this, SIGNAL(start_fist_track(std::vector<int>, int, bool, VirtualDevice::DEVICE_TYPE, bool, float, QString)),
            fist_track, SLOT(start(std::vector<int>, int, bool, VirtualDevice::DEVICE_TYPE, bool, float,QString)));
    // fist track start() sends err code
    connect(fist_track, SIGNAL(signal_start_failed(int)),
            this, SLOT(slot_start_failed(int)));
    // fist track start() exit
    connect(fist_track, SIGNAL(signal_start_exit()),
            this, SLOT(slot_start_exit()));
    // set label to show gesture kind
    connect(fist_track, SIGNAL(signal_set_label_gesture_kind(QString)),
            this, SLOT(slot_set_label_gesture_kind(QString)));
    // widegts:
    connect(spinBox_box_num_threads, SIGNAL(valueChanged(int)),
            this, SLOT(spinBox_box_num_threads_valueChanged(int)));      // box num threads
    connect(spinBox_kpt_num_threads, SIGNAL(valueChanged(int)),
            this, SLOT(spinBox_kpt_num_threads_valueChanged(int)));      // keypoint num threads
    connect(spinBox_camera_id, SIGNAL(valueChanged(int)),
            this, SLOT(spinBox_camera_id_valueChanged(int)));       // camera id
    connect(checkBox_show_result, SIGNAL(stateChanged(int)),
            this, SLOT(checkBox_show_result_stateChanged(int)));    // is show result
    connect(checkBox_start_stop, SIGNAL(stateChanged(int)),
            this, SLOT(checkBox_start_stop_stateChanged(int)));     // start or stop
#if ENABLE_VIRTUAL_CHOOSE
    connect(combobox_select_virtual_device, SIGNAL(currentIndexChanged(int)),
            this, SLOT(slot_combobox_select_virtual_device(int)));  // enable virtual mouse
#endif
#if ENABLE_CHOOSE_FILE
    connect(button_open_video_file, SIGNAL(clicked()),
            this, SLOT(slot_button_open_video_file()));             // get file name
#endif
    connect(doubleSpinBox_mouse_speed, SIGNAL(valueChanged(double)),
            this, SLOT(slot_doubleSpinBox_mouse_speed(double)));       // mouse speed
    connect(checkBox_enable_kalman_filter, SIGNAL(stateChanged(int)),
            this, SLOT(slot_checkBox_enable_kalman_filter(int)));   // enable kalman filter
}

Widget::~Widget()
{
    // destroyed the thread by signal and slot
    fist_track->terminate();
    fist_track_thread.quit();
    fist_track_thread.wait();

    delete ui;
}

/* rewrite the closeEvent(), when clicked the 'X' this function will be called */
void Widget::closeEvent(QCloseEvent *event)
{
    // otherwise when enable Show Result then click the 'X' directly,
    // the following error will occur:
    // "QPixmap::fromImage: QPixmap cannot be created without a QGuiApplication"
    // "QPixmap: Must construct a QGuiApplication before a QPixmap"
    this->checkBox_start_stop->setCheckState(Qt::Unchecked);
}

/* to set the image on QLabel object */
void Widget::load_cv_image(cv::Mat &cv_image)
{
    if (cv_image.empty())
        return;

    QImage qt_image = QImage((unsigned char*)(cv_image.data), cv_image.cols, cv_image.rows, QImage::Format_BGR888);
    this->label_result_image->setPixmap(
                QPixmap::fromImage(
//                   qt_image
                     qt_image.scaled(label_result_image->size(), Qt::KeepAspectRatio)
                    )
                );
}

/* set the gesture kind to label */
void Widget::slot_set_label_gesture_kind(QString str)
{
    QString prefix_str = "Gesture: ";
    this->label_gesture_kind->setText(prefix_str + str);
}

/* num thread changed SLOTS */
void Widget::spinBox_box_num_threads_valueChanged(int value)
{
    this->num_threads[0] = value;
}
void Widget::spinBox_kpt_num_threads_valueChanged(int value)
{
    this->num_threads[1] = value;
}

/* camera id changed SLOTS */
void Widget::spinBox_camera_id_valueChanged(int value)
{
    this->camera_id = value;
}

/* show result changed SLOTS */
void Widget::checkBox_show_result_stateChanged(int state)
{
    if (state == Qt::Checked) {
        this->show_result = true;
    } else if (state == Qt::Unchecked) {
        this->show_result = false;
    }
}

/* start Or stop changed SLOTS */
void Widget::checkBox_start_stop_stateChanged(int state)
{
    // Start
    if (state == Qt::Checked)
    {
        if (!fist_track_thread.isRunning())
        {
            fist_track_thread.start();  // start the thread not thread app(such as while...)
        }
        printf("[INFO] Current param:\n  num threads:%d, %d\n  camera id:%d\n  is show result:%d\n  "
               "[INFO] virtual device type:%d\n  enable kalman:%d\n  mouse speed:%1.1f\n  video path:%s\n",
               this->num_threads[0],
               this->num_threads[1],
               this->camera_id,
               this->show_result,
               this->virtual_device_type,
               this->enable_kalman_filter,
               this->mouse_speed,
               this->video_path.toStdString().c_str());
        // send signal to start the thread app (such as while ...)
        emit start_fist_track(this->num_threads,
                              this->camera_id,
                              this->show_result,
                              this->virtual_device_type,
                              this->enable_kalman_filter,
                              this->mouse_speed,
                              this->video_path);
    }
    // Terminate
    else if (state == Qt::Unchecked && fist_track_thread.isRunning())
    {
        fist_track->terminate();    // but the thread is not destroyed
    }
}

/* fist track start() failed process */
void Widget::slot_start_failed(int err_num)
{
    switch (err_num)
    {
    case -1:
        printf("[ERROR] ERROR CODE: Load NCNN model failed\n");
        break;
    case -2:
        printf("[ERROR] ERROR CODE: Open camera or video failed\n");
        break;
    case -3:
        printf("[ERROR] ERROR CODE: Open /device/uinput failed, try to use sudo\n");
    }
    // set checkBox_start_stop's state
    // Note:this will send the stateChanged signal too!
    this->checkBox_start_stop->setCheckState(Qt::Unchecked);
}

/* set the label's image as the default image */
void Widget::slot_start_exit()
{
    this->label_result_image->setPixmap(QPixmap(DEFAULT_IMAGE));
}
#if ENABLE_VIRTUAL_CHOOSE
/* enable virtual mouse slots */
void Widget::slot_combobox_select_virtual_device(int index)
{
    if (combobox_select_virtual_device->itemText(index) == COMBOBOX_ITEM_MOUSE) {
        this->virtual_device_type = VirtualDevice::VIRTUAL_MOUSE;
    }
    else if (combobox_select_virtual_device->itemText(index) == COMBOBOX_ITEM_KEYBOARD) {
        this->virtual_device_type = VirtualDevice::VIRTUAL_KEYBOARD;
    }
    else if (combobox_select_virtual_device->itemText(index) == COMBOBOX_ITEM_MOUSE_KEYBOARD) {
        this->virtual_device_type = VirtualDevice::VIRTUAL_MOUSE_KEYBOARD;
    }
    else {
        this->virtual_device_type = VirtualDevice::VIRTUAL_NONE;
    }
}
#endif
#if ENABLE_CHOOSE_FILE
void Widget::slot_button_open_video_file()
{
    // get file name
    this->video_path = QFileDialog::getOpenFileName(
                        this, tr("Open video file"), "",
                        tr("Files(*.mp4)"));
    label_current_file->setText(this->video_path);
}
#endif
void Widget::slot_doubleSpinBox_mouse_speed(double speed_ratio)
{
    this->mouse_speed = (float)speed_ratio;
}

void Widget::slot_checkBox_enable_kalman_filter(int state)
{
    if (state == Qt::Checked) {
        this->enable_kalman_filter = true;
    } else if (state == Qt::Unchecked) {
        this->enable_kalman_filter = false;
    }
}
