from pycocotools.coco import COCO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

def vis(img, boxes, kps=[]):
    for i in range(len(boxes)):
        box = boxes[i]
        box_x_min = int(box[0])
        box_y_min = int(box[1])
        box_w = int(box[2])
        box_h = int(box[3])
        color = (_COLORS[i] * 255).astype(np.uint8).tolist()

        cv2.rectangle(img, (box_x_min, box_y_min), (box_x_min + box_w, box_y_min + box_h), color, 2)

        if len(kps) > 0:
            kp_ls = np.array(kps[i]).reshape(-1, 3)[:, :2]
            for kp in kp_ls:
                cv2.circle(img, list(np.array(kp, int)), 2, color, 8)
    return img

if __name__ == "__main__":
    num_keypoints = 21
    train_or_val = "train"
    keypoints_json_file_name = f"keypoints_{num_keypoints}_{train_or_val}.json"
    images_out_dir = f"out/coco_c430_hand_keypoints_v2/images/{train_or_val}"
    annotations_out_dir = "out/coco_c430_hand_keypoints_v2/annotations"

    coco = COCO(os.path.join(annotations_out_dir, keypoints_json_file_name))
    img_ids = coco.getImgIds()
    for img_id in img_ids:
        img_obj = coco.loadImgs(img_id)[0]
        img = cv2.imread(os.path.join(images_out_dir, img_obj["file_name"]))
        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann_objs = coco.loadAnns(ann_ids)
        bboxes = []
        kpts = []
        for ann in ann_objs:
            bboxes.append(ann["bbox"])
            kpts.append(ann["keypoints"])
        img = vis(img, bboxes, kpts)
        cv2.namedWindow("result", 0)
        cv2.resizeWindow("result", 1280, 720)
        cv2.imshow("result", img)
        cv2.waitKey(-1)