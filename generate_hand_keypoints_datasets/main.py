import json
import os
import datetime
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm


mp_hands = mp.solutions.hands
hand_keypoints_mask = range(21)
# hand_keypoints_mask = [0, 4, 8, 12, 16, 20]
num_keypoints = len(hand_keypoints_mask)
train_or_val = "val"
save_image = True
video_files_root_path = os.path.join("in", train_or_val)
keypoints_json_file_name = f"keypoints_{num_keypoints}_{train_or_val}.json"
interval = 4    # 每几帧截取一张图片
# 输出路径
images_out_dir = f"out/coco_c430_hand_keypoints/images/{train_or_val}"
annotations_out_dir = "out/coco_c430_hand_keypoints/annotations"
years_month_hours = datetime.datetime.now().strftime("%Y%m%d-%H")

"""
    1. 读取视频帧
    2. mediapipe检测手部关键点
    3. 根据关键点生成外接矩形作为bbox
    4. 将关键点和bbox写入到coco格式的json文件中
"""
"""
    类别id, 图片id, 标签id
"""


def get_max_id(info_list):
    max_id = 0
    for info in info_list:
        max_id = max(max_id, info["id"])
    return max_id


""" 求外接矩形作为边界框 """


def get_bbox_from_keypoints_scale(kpts, img_w, img_h, scale=1.2):
    kpts_np = np.array(kpts)
    # 获得关键点为(x,y,2)的mask，排除(0,0,0)的干扰
    mask = kpts_np[2::3] == 2
    # 获得外接矩形左上角顶点和右下角顶点
    kpts_xmin = min(kpts_np[0::3][mask])
    kpts_ymin = min(kpts_np[1::3][mask])
    kpts_xmax = max(kpts_np[0::3][mask])
    kpts_ymax = max(kpts_np[1::3][mask])
    # 外接矩形按比例扩大, 保证扩张后的外接矩形不超出图像区域
    bbox_xcenter = (kpts_xmin + kpts_xmax) / 2
    bbox_ycenter = (kpts_ymin + kpts_ymax) / 2
    bbox_w = (kpts_xmax - kpts_xmin) * scale
    bbox_h = (kpts_ymax - kpts_ymin) * scale
    bbox_xmin = max(0, bbox_xcenter - bbox_w / 2)
    bbox_xmax = min(img_w, bbox_xcenter + bbox_w / 2)
    bbox_ymin = max(0, bbox_ycenter - bbox_h / 2)
    bbox_ymax = min(img_h, bbox_ycenter + bbox_h / 2)
    bbox_w = bbox_xmax - bbox_xmin
    bbox_h = bbox_ymax - bbox_ymin
    # 从np.int32变为int，否则写入json会报错
    return int(bbox_xmin), int(bbox_ymin), int(bbox_w), int(bbox_h)


def get_bbox_from_keypoints(kpts, img_w, img_h):
    kpts_np = np.array(kpts)
    # 获得关键点为(x,y,2)的mask，排除(0,0,0)的干扰
    mask = kpts_np[2::3] == 2
    kpts_xmin = max(0, min(kpts_np[0::3][mask]) - 10)
    kpts_ymin = max(0, min(kpts_np[1::3][mask]) - 10)
    kpts_xmax = min(img_w, max(kpts_np[0::3][mask]) + 10)
    kpts_ymax = min(img_h, max(kpts_np[1::3][mask]) + 10)
    bbox_xmin = kpts_xmin
    bbox_ymin = kpts_ymin
    bbox_w = kpts_xmax-kpts_xmin
    bbox_h = kpts_ymax-kpts_ymin
    # 从np.int32变为int，否则写入json会报错
    return int(bbox_xmin), int(bbox_ymin), int(bbox_w), int(bbox_h)


def get_annotations(img, available_images_id, available_annotations_id):
    annotations_list = []

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:

        # 读取一张图片的信息
        image_height = img.shape[0]
        image_width = img.shape[1]

        img_o = img.copy()
        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)

        # 检测图片手部关键点
        results = hands.process(img_o)
        # 没有结果则直接返回
        if not results.multi_hand_landmarks:
            return annotations_list, available_annotations_id
        # 遍历每个手
        for hand_landmarks in results.multi_hand_landmarks:
            kpts = []
            # 遍历每个关键点
            for i in range(21):
                # 记录每个手的关键点，即使有手超出图片也有坐标，坐标会超出图片大小
                kpt_x = int(hand_landmarks.landmark[i].x * image_width)
                kpt_y = int(hand_landmarks.landmark[i].y * image_height)
                # 如果关键点超出图片大小，那么置为[0,0,0]
                if kpt_x < 0 or kpt_y < 0 or kpt_x > image_width or kpt_y > image_height:
                    kpts.extend([0, 0, 0])
                # 图片内的关键点
                else:
                    kpts.extend([kpt_x, kpt_y, 2])
            bbox_xmin, bbox_ymin, bbox_w, bbox_h = get_bbox_from_keypoints_scale(
                                                    kpts, image_width, image_height, scale=1.13)
            bbox = [bbox_xmin, bbox_ymin, bbox_w, bbox_h]
            save_kpts = []
            for idx in hand_keypoints_mask:
                save_kpts += kpts[idx*3: idx*3+3]
            annotations_list.append({
                'area': bbox_w * bbox_h,
                'bbox': bbox,
                'category_id': 0,
                'id': available_annotations_id,
                'image_id': available_images_id,
                'iscrowd': 0,
                'keypoints': save_kpts,     # 取关键点
                'num_keypoints': num_keypoints
            })
            available_annotations_id += 1

    return annotations_list, available_annotations_id


""" 处理视频：保存视频片段、返回片段中对应的目标信息 """


def handle_video_file(video_path, available_images_id, available_annotations_id, interval=3):
    video_img_anno_info = {'annotations': [], 'images': []}
    if os.path.isfile(video_path) and (os.path.splitext(video_path)[-1] in [".mp4", ".MOV", ".mov"]):
        video = cv2.VideoCapture(video_path)
        frame_cnt = 0
        while video.isOpened():
            ret, img = video.read()
            if not ret:
                break
            img_h = img.shape[0]
            img_w = img.shape[1]
            frame_cnt += 1
            if frame_cnt % interval == 0:
                annotations_list, available_annotations_id = get_annotations(
                    img, available_images_id, available_annotations_id)
                # 如果存在目标，则保存图片和对应的annotations
                if len(annotations_list) > 0:
                    # 保存图片和信息
                    save_img_name = "{}-{}-{}.jpg".format(
                        years_month_hours, train_or_val, available_images_id)
                    if save_image:
                        cv2.imwrite(os.path.join(
                            images_out_dir, save_img_name), img)
                    video_img_anno_info["images"].append({"file_name": save_img_name,
                                                          "id": available_images_id,
                                                          "width": img_w,
                                                          "height": img_h})
                    available_images_id += 1
                    # 保存标注信息
                    video_img_anno_info["annotations"].extend(annotations_list)
    else:
        print(f"File:{video_path} is not a video, skip")
    return video_img_anno_info, available_images_id, available_annotations_id


if __name__ == "__main__":
    json_data = None
    max_images_id = 0
    max_categories_id = 0
    max_annotations_id = 0

    if not os.path.exists(images_out_dir):
        os.makedirs(images_out_dir)
    if not os.path.exists(annotations_out_dir):
        os.makedirs(annotations_out_dir)
    """ 如果存在json文件，在原来的基础上追加 """
    if (os.path.exists(keypoints_json_file_name)):
        with open(keypoints_json_file_name, "r") as fp:
            json_data = json.load(fp)

        " 获取当前最大的images_id, categories_id, annotations_id "
        max_images_id = get_max_id(json_data["images"])
        max_categories_id = get_max_id(json_data["categories"])
        max_annotations_id = get_max_id(json_data["annotations"])

        print(max_images_id, max_categories_id, max_annotations_id)
    else:
        json_data = {'categories': [], 'annotations': [], 'images': []}
        json_data["categories"] = [
            {"id": 0,
             "name": "hand",
             "supercategory": "person",
             "keypoints": [
                 "WRIST",
                 "THUMB_CMC",
                 "THUMB_MCP",
                 "THUMB_IP",
                 "THUMB_TIP",
                 "INDEX_FINGER_MCP",
                 "INDEX_FINGER_PIP",
                 "INDEX_FINGER_DIP",
                 "INDEX_FINGER_TIP",
                 "MIDDLE_FINGER_MCP",
                 "MIDDLE_FINGER_PIP",
                 "MIDDLE_FINGER_DIP",
                 "MIDDLE_FINGER_TIP",
                 "RING_FINGER_MCP",
                 "RING_FINGER_PIP",
                 "RING_FINGER_DIP",
                 "RING_FINGER_TIP",
                 "PINKY_MCP",
                 "PINKY_PIP",
                 "PINKY_DIP",
                 "PINKY_TIP",
             ],
                "skeleton": [[ 0,  1],
                            [ 1,  2],
                            [ 2,  3],
                            [ 3,  4],
                            [ 0,  5],
                            [ 5,  6],
                            [ 6,  7],
                            [ 7,  8],
                            [ 5,  9],
                            [ 9, 10],
                            [10, 11],
                            [11, 12],
                            [ 9, 13],
                            [13, 14],
                            [14, 15],
                            [15, 16],
                            [13, 17],
                            [ 0, 17],
                            [17, 18],
                            [18, 19],
                            [19, 20]]
             }]

    """ 读取视频文件 """
    available_images_id = max_images_id + 1
    current_categories_id = max_categories_id + 1
    available_annotations_id = max_annotations_id + 1

    video_files_names = os.listdir(video_files_root_path)

    for video_file_name in video_files_names:
        print(f"Process file: {video_file_name}...")
        video_path = os.path.join(video_files_root_path, video_file_name)
        # 保存视频中的帧，并返回这个视频中的目标信息
        video_img_anno_info, available_images_id, available_annotations_id = handle_video_file(video_path, available_images_id,
                                                                                               available_annotations_id, interval)
        # 如果视频中存在目标则保存
        if (len(video_img_anno_info["images"]) > 0):
            json_data["images"].extend(video_img_anno_info["images"])
            json_data["annotations"].extend(video_img_anno_info["annotations"])
    images_num = len(json_data["images"])
    annotations_num = len(json_data["annotations"])
    print(f"Total images: {images_num}, annotations: {annotations_num}")
    with open(os.path.join(annotations_out_dir, keypoints_json_file_name), "w") as fp:
        json.dump(json_data, fp)
