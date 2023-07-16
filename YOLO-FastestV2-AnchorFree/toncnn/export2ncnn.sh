#! /bin/bash
set -v
python3 -m onnxsim YOLO-FastestV2-AnchorFree.onnx YOLO-FastestV2-AnchorFree-opt.onnx
./onnx2ncnn YOLO-FastestV2-AnchorFree-opt.onnx YOLO-FastestV2-AnchorFree.param YOLO-FastestV2-AnchorFree.bin
./ncnnoptimize YOLO-FastestV2-AnchorFree.param YOLO-FastestV2-AnchorFree.bin YOLO-FastestV2-AnchorFree-opt.param YOLO-FastestV2-AnchorFree-opt.bin 1