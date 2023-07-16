import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import Lite_HRNet_SimCC
from draw_utils import draw_keypoints
import transforms


def predict_all_hand():
    # TODO
    pass


def predict_single_hand():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    flip_test = False
    resize_hw = (128, 128)
    img_path = ""   # 图片路径
    weights_path = ""   # 模型权重路径
    keypoint_json_path = "hand_21_keypoints.json"
    simcc_split_ratio = 2.0
    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.1, 1.1), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read json file
    with open(keypoint_json_path, "r") as f:
        hand_info = json.load(f)

    # read single-target image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # create model
    model = Lite_HRNet_SimCC(num_joints=21, input_size=[resize_hw[0], resize_hw[1]], simcc_split_ratio=simcc_split_ratio)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    with torch.inference_mode():
        pred_x, pred_y = model(img_tensor.to(device))

        if flip_test:
            # TODO: no filp test code yet!
            pass

        keypoints, scores = transforms.get_final_preds(pred_x, pred_y, simcc_split_ratio, [target["reverse_trans"]])
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)

        plot_img = draw_keypoints(img, keypoints, scores, thresh=0.04, r=5)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save("test_result.jpg")


if __name__ == '__main__':
    predict_single_hand()
