import os
import json
import time
import torch
import cv2
import numpy as np

from build_utils import img_utils, torch_utils, utils
from build_utils.parse_config import parse_data_cfg
from build_utils.utils import *
from models import detector
from draw_box_utils import draw_box


def main():
    img_size = 352  # 必须是32的整数倍 [352, 416, 512, 608]
    data = "data/mydata.data"
    weights = "weights/best.pt"  # 训练好的权重文件
    json_path = "data/classes.json"  # json标签文件
    video_path = "../../datasets/test_video/CW.mp4"
    assert os.path.exists(data), "data file {} dose not exist.".format(data)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    assert os.path.exists(video_path), "video file {} dose not exist.".format(video_path)

    data_dict = parse_data_cfg(data)
    # 类型转换
    data_dict["classes"] = int(data_dict["classes"])
    data_dict["anchor_num"] = int(data_dict["anchor_num"])
    
    original_video = cv2.VideoCapture(video_path)
    video_width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    video_height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    video_fps = original_video.get(cv2.CAP_PROP_FPS)
    video_out_path = "out/" + "out_" + os.path.basename(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out_path, fourcc, video_fps, (video_width, video_height), True)

    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = detector.Detector(data_dict["classes"], data_dict["anchor_num"], init_backbone=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.to(device)

    model.eval()
    with torch.no_grad():
        # init
        img = torch.zeros((1, 3, img_size, img_size), device=device)
        model(img)

        while original_video.isOpened():
            ret, img_o = original_video.read()  # BGR
            if not ret:
                break

            img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 dim
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).float()
            img /= 255.0  # scale (0, 255) to (0, 1)
            img = img.unsqueeze(0)  # add batch dimension
            img -= torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype, device=img.device).reshape(1, 3, 1, 1)
            img /= torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype, device=img.device).reshape(1, 3, 1, 1)

            t1 = torch_utils.time_synchronized()
            outputs = model(img)
            t2 = torch_utils.time_synchronized()
            print("interface:", t2 - t1)

            output_boxes = non_max_suppression(outputs, conf_thres=0.3, iou_thres=0.3)[0]
            t3 = time.time()
            print("nms:", t3 - t2)

            if output_boxes is None:
                print("No target detected.")
                writer.write(img_o)
                continue

            # process detections
            output_boxes[:, :4] = utils.scale_coords(img.shape[2:], output_boxes[:, :4], img_o.shape).round()
            print(output_boxes.shape)

            bboxes = output_boxes[:, :4].detach().cpu().numpy()
            scores = output_boxes[:, 4].detach().cpu().numpy()
            classes = output_boxes[:, 5].detach().cpu().numpy().astype(np.int64) + 1

            img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
            img_o = cv2.cvtColor(np.asarray(img_o), cv2.COLOR_RGB2BGR)
            writer.write(img_o)

        writer.release()


if __name__ == "__main__":
    main()
