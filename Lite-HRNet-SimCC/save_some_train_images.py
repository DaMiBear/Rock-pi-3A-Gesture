# Plotting utils

import glob
import math
import os
import random
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont


import torch.nn as nn
from torch.utils import data
import os
import transforms
from my_dataset_coco import CocoKeypoint
import json
# Settings
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        self.palette = [self.hex2rgb(c) for c in matplotlib.colors.TABLEAU_COLORS.values()]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

def cxcywh2xyxy(bboxes):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # top left x
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # top left y
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # bottom right x
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # bottom right y
    return bboxes

def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

def plot_one_box(x, im, color=None, label=None, line_thickness=3, kpt_label=False, kpts=None, kpts_vis=None,steps=2, orig_shape=None):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(im, c1, c2, (255,0,0), thickness=tl*2//3, lineType=cv2.LINE_AA)
    # if label:
    #     if len(label.split(' ')) > 1:
    #         label = label.split(' ')[-1]
    #         tf = max(tl - 1, 1)  # font thickness
    #         t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
    #         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #         cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
    #         cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)
    if kpt_label:
        plot_skeleton_kpts(im, kpts, kpts_vis, steps, orig_shape=orig_shape)


def plot_skeleton_kpts(im, kpts, kpts_vis, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [25, 255, 255], [25, 25, 255]], dtype=np.uint8)

    skeleton = [[ 0,  1],
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

    # pose_limb_color = palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]]
    pose_limb_color = palette[[16 for _ in range(kpts.shape[0])]]
    # pose_kpt_color = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]
    pose_kpt_color = palette[[18 for _ in range(kpts.shape[0])]]

    radius = 3
    num_kpts = kpts.shape[0]

    for kid in range(num_kpts):
        if kpts_vis[kid] == 0:
            continue
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[kid][0], kpts[kid][1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        if kpts_vis[sk[0]] == 0 or kpts_vis[sk[1]] == 0:
            continue
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[sk[0]][0]), int(kpts[sk[0]][1]))
        pos2 = (int(kpts[sk[1]][0]), int(kpts[sk[1]][1]))
        
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


def plot_one_box_PIL(box, im, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image 'im' using PIL
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    line_thickness = line_thickness or max(int(min(im.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=tuple(color))  # plot
    if label:
        fontsize = max(round(max(im.size) / 40), 12)
        font = ImageFont.truetype("Arial.ttf", fontsize)
        txt_width, txt_height = font.getsize(label)
        draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(color))
        draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(im)



def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        kpts = o[:,6:]
        o = o[:,:6]
        for index, (*box, conf, cls) in enumerate(o.cpu().numpy()):
            targets.append([i, cls, *list(*xyxy2cxcywh(np.array(box)[None])), conf, *list(kpts.cpu().numpy()[index])])
    return np.array(targets)


def save_some_train_images(images, targets, paths=None, fname='images.png', names=None, max_size=640, max_subplots=16, kpt_label=False, steps=2, orig_shape=None):
    # Plot image grid with labels
    # tatgets[bs, max_label_num, (class, xc,yc,w,h, kptx,kpty,...) ]
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        # images *= np.array([0.229, 0.224, 0.225], dtype=images.dtype).reshape(1, 3, 1, 1)
        # images += np.array([0.485, 0.456, 0.406], dtype=images.dtype).reshape(1, 3, 1, 1)
        images *= 255.0

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[i]
            num_keypoints = image_targets["keypoints"].shape[0]
            labels = True   # 直接赋值为True，因为不是网络预测的结果，而是真实标签
            classes = [ 0 for _ in range(len(image_targets))]
            conf = None if labels else image_targets["score"]  # check for confidence presence (label vs pred)
            if kpt_label:
                kpts = image_targets["keypoints"]   # kpts shape: (num_kpts, 2)
                kpts_vis = image_targets["visible"]
            else:
                kpts = None

            # if boxes.shape[1]:
            #     if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
            #         boxes[[0, 2]] *= w  # scale to pixels
            #         boxes[[1, 3]] *= h
            #     elif scale_factor < 1:  # absolute coords need scale if image scales
            #         boxes *= scale_factor
            # boxes[[0, 2]] += block_x
            # boxes[[1, 3]] += block_y

            if kpt_label and (num_keypoints > 0):
                if kpts.max()<1.01:
                    kpts[:, 0] *= w # scale to pixels
                    kpts[:, 1] *= h
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    kpts *= scale_factor
                kpts[:, 0] += block_x
                kpts[:, 1] += block_y
            
            plot_one_box(None, mosaic, label=None, color=colors(0), line_thickness=tl, kpt_label=kpt_label, kpts=kpts, kpts_vis=kpts_vis, steps=steps, orig_shape=orig_shape)

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]

            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 6, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))  # cv2 save
        # mosaic = mosaic[:,:,::-1]
        # Image.fromarray(mosaic).save(fname)  # PIL save
    return mosaic



def plot_labels(labels, names=(), save_dir=Path(''), loggers=None):
    # plot dataset labels
    print('Plotting labels... ')
    c, b, kpts = labels[:, 0], labels[:, 1:5].transpose(), labels[:, 5:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # seaborn correlogram
    sns.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use('svg')  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sns.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sns.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = cxcywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000, :5]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()

    # loggers
    for k, v in loggers.items() or {}:
        if k == 'wandb' and v:
            v.log({"Labels": [v.Image(str(x), caption=x.name) for x in save_dir.glob('*labels*.jpg')]}, commit=False)

def filter_out_bound_kpts(keypoints, keypoints_vis, image_size):
    '''
    :param keypoints:  [num_keypoints, 2]
    :param keypoints_vis: [num_keypionts,]
    :param image_size: image_size (w, h)
    :return: keypoints, keypoints_weights(>0: visible, 0: invisible)
    '''
    num_keypoints = keypoints.shape[0]
    keypoints_weights = np.ones((num_keypoints), dtype=np.float32)
    keypoints_weights[keypoints_vis == 0] = 0     # 保留原来不可见的点

    # detect abnormal coords and make the weight 0
    for kpt_id in range(num_keypoints):
        # y < 0
        if keypoints[kpt_id][1] < 0:
            keypoints_weights[kpt_id] = 0
            keypoints_vis[kpt_id] = 0
            keypoints[kpt_id][1] = 0
        # y >= image height
        elif keypoints[kpt_id][1] >= image_size[0]:
            keypoints_weights[kpt_id] = 0
            keypoints_vis[kpt_id] = 0
            keypoints[kpt_id][1] = image_size[0] - 1
        # x < 0
        if keypoints[kpt_id][0] < 0:
            keypoints_weights[kpt_id] = 0
            keypoints_vis[kpt_id] = 0
            keypoints[kpt_id][0] = 0
        # x > image width
        elif keypoints[kpt_id][0] >= image_size[1]:
            keypoints_weights[kpt_id] = 0
            keypoints_vis[kpt_id] = 0
            keypoints[kpt_id][0] = image_size[1] - 1

    return keypoints, keypoints_vis, keypoints_weights

if __name__ == "__main__":
    keypoints_path = "hand_21_keypoints.json"
    fixed_size = (128, 128)
    data_root = ""   # 数据集路径
    batch_size = 32
    nw = 8
    max_iter = 10
    out_dir = "images_outputs"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(keypoints_path, "r") as f:
        kps_info = json.load(f)

    data_transform = {
        "train": transforms.Compose([
            transforms.AffineTransform(
                scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            transforms.RandomHorizontalFlip(0.5, kps_info["flip_pairs"]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            #                      0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(
                scale=(1.25, 1.25), fixed_size=fixed_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            #                      0.229, 0.224, 0.225])
        ])
    }

    train_dataset = CocoKeypoint(data_root, "train", transforms=data_transform["train"], fixed_size=fixed_size)
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=False,
                                        num_workers=nw,
                                        collate_fn=train_dataset.collate_fn)

    for i, [images, targets] in enumerate(train_data_loader):
        if i > max_iter:
            break
        # 因为数据增强中去掉了KeypointToSimCC（包括关键点超出增强后图片范围滤除的代码）
        # 所以这里额外增加滤除关键点的代码
        for t in targets:
            kpts = t["keypoints"]
            kpts_vis = t["visible"]
            t["keypoints"], t["visible"], _ = filter_out_bound_kpts(t["keypoints"], t["visible"], fixed_size)
        f = os.path.join(out_dir, f'train_batch{i}.png')  # filename
        save_some_train_images(images, targets, fname=f, kpt_label=True)

