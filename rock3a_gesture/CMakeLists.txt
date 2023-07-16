cmake_minimum_required(VERSION 3.5)

project(rock3a_gesture VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")

# set(CMAKE_FIND_ROOT_PATH "/usr/aarch64-linux-gnu")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_FLAGS "-march=armv8-a" )
set(CMAKE_CXX_FLAGS "-march=armv8-a")

set(Qt5_DIR "/home/dm-wsl/aarch64_lib/qt5.15.5/lib/cmake/Qt5")
set(NumCpp_DIR "/home/dm-wsl/aarch64_lib/numcpp/share/NumCpp/cmake")
set(ncnn_DIR "/home/dm-wsl/aarch64_lib/ncnn/lib/cmake/ncnn")
set(OpenCV_DIR "/home/dm-wsl/aarch64_lib/opencv-aarch64-linux-4.5.4/lib/cmake/opencv4")

find_package(Qt5 COMPONENTS Widgets Concurrent Test REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)
find_package(NumCpp 2.6.2 REQUIRED)

set(PROJECT_SOURCES
        main.cpp
        widget.cpp
        fist_track.cpp
        fist_track.h
        yolo-fastestv2-anchorfree.cpp
        yolo-fastestv2-anchorfree.h
        linux_virtual_device.cpp
        linux_virtual_device.h
        kalman_filter.cpp
        kalman_filter.h
        keypointsdetector.h keypointsdetector.cpp
        widget.h
        widget.ui
        src.qrc
)


add_executable(rock3a_gesture
    ${PROJECT_SOURCES}
)


target_link_libraries(rock3a_gesture PRIVATE Qt5::Widgets)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE NumCpp::NumCpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ncnn)
