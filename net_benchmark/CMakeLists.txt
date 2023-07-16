cmake_minimum_required (VERSION 3.14)

project(net_interface_benchmark)

set(ncnn_DIR "/home/dm-wsl/aarch64_lib/ncnn/lib/cmake/ncnn")

find_package(ncnn REQUIRED)

message(${PROJECT_NAME})
set(PROJECT_SOURCES
    net_interface_benchmark.cpp
)
add_executable(${PROJECT_NAME}  ${PROJECT_SOURCES})

target_link_libraries(${PROJECT_NAME}  ncnn)