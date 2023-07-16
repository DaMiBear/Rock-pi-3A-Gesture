import torch
from build_utils.datasets_coco import LoadImagesAndLabels_COCO
from build_utils.datasets import LoadImagesAndLabels
import numpy as np
import cv2
import yaml
from build_utils.parse_config import parse_data_cfg

def draw_batch(mini_batch):
    imgs = mini_batch[0]
    batch_size = imgs.shape[0]
    imgs_width = imgs.shape[2]
    imgs_height = imgs.shape[3]
    labels = mini_batch[1].numpy().astype(np.float64)  # 个数不一定对应batch size [batch_index, cls, x_center, y_center, width, height] normalized(0-1)
    
    for batch_index in range(batch_size):
        img = imgs[batch_index].permute(1, 2, 0).numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask = labels[:, 0] == batch_index
        labels_per_img = labels[mask, 1:]
        """ 绘制边界框 """
        for l in labels_per_img:
            cls = l[0].astype(np.int64)
            bbox = l[1:]
            cv2.circle(img, center=(int(bbox[0] * imgs_width) , int(bbox[1] * imgs_height)), radius=4, color=(0, 255, 0), thickness=3)
            # [x_center, y_center, width, height] -> [l, t, r, b] 图片像素尺度
            bbox[0] = (bbox[0] - bbox[2] / 2) * imgs_width     # x_min
            bbox[1] = (bbox[1] - bbox[3] / 2) * imgs_height    # y_min
            bbox[2] = bbox[0] + bbox[2] * imgs_width        # x_max = x_min + width
            bbox[3] = bbox[1] + bbox[3] * imgs_height       # y_max = y_min + height
            bbox = bbox.astype(np.int64)
            cv2.rectangle(img, bbox[0:2], bbox[2:4], color=(255, 0, 0), thickness=3)
        cv2.imwrite(f"batch{batch_index}.jpg", img)

if __name__ == "__main__":
    data_path = "data/mydata.data"
    bs = 8
    with open("cfg/hyp.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    data_dict = parse_data_cfg(data_path)
    data_dict["classes"] = int(data_dict["classes"])
    data_dict["anchor_num"] = int(data_dict["anchor_num"])

    train_dataset = LoadImagesAndLabels_COCO(path=data_dict["dataset_dir"],
                                             images_dir=data_dict["train_images_dir"],
                                             json_file=data_dict["train_json_file"],
                                             img_size=352,
                                             batch_size=bs, augment=True, hyp=hyp, 
                                             cache_images=False)
    # train_dataset = LoadImagesAndLabels(path="../datasets/yolo_c430_lite/train.txt",
    #                                     img_size=352,
    #                                     batch_size=bs, augment=True, hyp=hyp, 
    #                                     cache_images=False,
    #                                     )
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=bs,
                                                   num_workers=8,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)
    batch_data = next(iter(train_dataloader))
    draw_batch(batch_data)

