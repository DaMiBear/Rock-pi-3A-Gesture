import math
import os
import random
import shutil

import cv2
import numpy as np
import torch
from PIL import ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from pycocotools.coco import COCO

from build_utils.utils import xyxy2xywh, xywh2xyxy

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

# get orientation in exif tag
# 找到图像exif信息中对应旋转信息的key值
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def exif_size(img):
    """
    获取图像的原始img size
    通过exif的orientation信息判断图像是否有旋转，如果有旋转则返回旋转前的size
    :param img: PIL图片
    :return: 原始图像的size
    """
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270  顺时针翻转90度
            s = (s[1], s[0])
        elif rotation == 8:  # ratation 90  逆时针翻转90度
            s = (s[1], s[0])
    except:
        # 如果图像的exif信息中没有旋转信息，则跳过
        pass

    return s


class LoadImagesAndLabels_COCO(Dataset):  # for training/testing

    def __init__(
            self,
            path,  # coco数据集根路径
            images_dir,  # 图片文件夹路径
            json_file,  # 标签json文件名
            # 这里设置的是预处理后输出的图片尺寸
            # 当为训练集时，设置的是训练过程中(开启多尺度)的最大尺寸
            # 当为验证集时，设置的是最终使用的网络大小
            img_size=325,
            batch_size=32,
            augment=False,  # 训练集设置为True(augment_hsv)，验证集设置为False
            hyp=None,  # 超参数字典，其中包含图像增强会使用到的超参数
            rect=False,  # 是否使用rectangular training
            cache_images=False,  # 是否缓存图片到内存中
            single_cls=False,
            pad=0.0,
            rank=-1):

        self.json_file = os.path.join(path, "annotations", json_file)
        try:
            # 读取图片路径到列表中
            if os.path.isfile(self.json_file):
                self.coco = COCO(self.json_file)
                # 读取每个图片路径信息
                self.ids = self.coco.getImgIds()    # 图片Id
                self.imgs_ann = self.coco.loadImgs(self.ids)
                f = [os.path.join(path, images_dir, img_path["file_name"]) for img_path in self.imgs_ann]
            else:
                raise Exception("%s does not exist" % self.json_file)

            # 检查每张图片后缀格式是否在支持的列表中，保存支持的图像路径
            # img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
            self.img_files = [
                x for x in f if os.path.splitext(x)[-1].lower() in img_formats
            ]

        except Exception as e:
            raise FileNotFoundError("Error loading data from {}. {}".format(
                path, e))

        # 如果图片列表中没有图片，则报错
        n = len(self.img_files)
        assert n > 0, "No images found in %s." % (os.path.join(path, images_dir))

        # batch index
        # 将数据划分到一个个batch中
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        # 记录数据集划分后的总batch数
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images 图像总数目
        self.batch = bi  # batch index of image 记录哪些图片属于哪个batch
        self.img_size = img_size  # 这里设置的是预处理后输出的图片尺寸
        self.augment = augment  # 是否启用augment_hsv
        self.hyp = hyp  # 超参数字典，其中包含图像增强会使用到的超参数
        self.rect = rect  # 是否使用rectangular training
        # 注意: 开启rect后，mosaic就默认关闭
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.stop_mosaic = False  # 用于训练时最后几个epoch停止mosaic

        # 记录每张图像的原始尺寸
        # self.shapes = []
        # for ann in self.imgs_ann:
        #     self.shapes += []
        self.shapes = np.array([ [ann["width"], ann["height"]] for ann in self.imgs_ann], dtype=np.float64)

        # labels: [class, x_min, y_min, width, height] 绝对像素值
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n
        nm, nf, ne, nd = 0, 0, 0, 0  # number mission, found, empty, duplicate

        # 处理进度条只在第一个进程中显示
        if rank in [-1, 0]:
            pbar = tqdm(self.ids)
        else:
            pbar = self.ids

        # 遍历载入标签文件
        for i, img_id in enumerate(pbar):
            # 读取标签信息
            try:
                anno_ids = self.coco.getAnnIds(img_id)
                img_width_o = self.coco.loadImgs(img_id)[0]["width"]
                img_height_o = self.coco.loadImgs(img_id)[0]["height"]
                # l shape[num_label_per_img, 5(cls, x_min, y_min, width, height)]，0-1归一化
                l = np.array([[anno["category_id"], 
                               anno["bbox"][0] / img_width_o,
                               anno["bbox"][1] / img_height_o,
                               anno["bbox"][2] / img_width_o,
                               anno["bbox"][3] / img_height_o] for anno in self.coco.loadAnns(anno_ids)],
                    dtype=np.float32)
                # 因为coco的bbox格式是[x_min, y_min, width, height]，yolo格式为[x_center, y_center, width, height]
            except Exception as e:
                print("An error occurred while loading the coco label")
                nm += 1  # file missing
                continue

            # 如果标注信息不为空的话
            if l.shape[0]:
                # 标签信息每行必须是五个值[class, x_min, y_min, width, height]
                assert l.shape[1] == 5, "> 5 label columns img_id: %s" % img_id
                assert (l >= 0).all(), "negative labels img_id: %s" % img_id
                assert (l[:, 1:] <= 1).all(
                ), "labels has out bound data img_id: %s" % img_id

                # 检查每一行，看是否有重复信息
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode

                self.labels[i] = l
                nf += 1  # file found

            else:
                ne += 1  # file empty

            # 处理进度条只在第一个进程中显示
            if rank in [-1, 0]:
                # 更新进度条描述信息
                pbar.desc = "Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)" % (
                    nf, nm, ne, nd, n)
        assert nf > 0, "No labels found in coco datasets."

        # Rectangular Training https://github.com/ultralytics/yolov3/issues/232
        # 如果为ture，训练网络时，会使用类似原图像比例的矩形(让最长边为img_size)，而不是img_size x img_size
        # 注意: 开启rect后，mosaic就默认关闭
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            # 计算每个图片的高/宽比
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            # argsort函数返回的是数组值从小到大的索引值
            # 按照高宽比例进行排序，这样后面划分的每个batch中的图像就拥有类似的高宽比
            irect = ar.argsort()
            # 根据排序后的顺序重新设置图像顺序、标签顺序以及shape顺序
            self.img_files = [self.img_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # set training image shapes
            # 计算每个batch采用的统一尺度
            shapes = [[1, 1]] * nb  # nb: number of batches
            for i in range(nb):
                ari = ar[bi == i]  # bi: batch index
                # 获取第i个batch中，最小和最大高宽比
                mini, maxi = ari.min(), ari.max()

                # 如果高/宽小于1(w > h)，将w设为img_size
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                # 如果高/宽大于1(w < h)，将h设置为img_size
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            # 计算每个batch输入网络的shape值(向上设置为32的整数倍)
            self.batch_shapes = np.ceil(
                np.array(shapes) * img_size / 32. + pad).astype(np.int) * 32
        
        self.imgs = [None] * n  # n为图像总数
        # Cache images into memory for faster training (Warning: large datasets may exceed system RAM)
        if cache_images:  # if training
            gb = 0  # Gigabytes of cached images 用于记录缓存图像占用RAM大小
            if rank in [-1, 0]:
                pbar = tqdm(range(len(self.img_files)), desc="Caching images")
            else:
                pbar = range(len(self.img_files))

            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(
                    self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes  # 用于记录缓存图像占用RAM大小
                if rank in [-1, 0]:
                    pbar.desc = "Caching images (%.1fGB)" % (gb / 1E9)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        hyp = self.hyp
        if self.mosaic:
            if self.stop_mosaic is False:
                # load mosaic
                # 返回图片大小为网络输入大小self.img_size
                # labels是绝对坐标xyxy形式(self.img_size范围内)
                img, labels = load_mosaic(self, index)
            else:
                img, (h0, w0), (h, w) = load_image(
                    self, index)  # 把图片最长边reshape到self.img_size保持宽高比不变
                img, ratio, pad = letterbox(img, self.img_size, auto=False)
                labels = []
                x = self.labels[index]
                if x.size > 0:
                    # Normalized xywh to pixel xyxy format
                    labels = x.copy()  # label: [class, x_min, y_min, width, height]
                    # xywh->xyxy大小为输入尺度
                    labels[:, 1] = ratio[0] * w * x[:, 1] + pad[0]  # pad width
                    labels[:, 2] = ratio[1] * h * x[:, 2] + pad[1]  # pad height
                    labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3]) + pad[0]
                    labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4]) + pad[1]

            shapes = None  # 训练模式用不上shapes
        else:
            # load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # letterbox
            shape = self.batch_shapes[self.batch[
                index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img,
                                        shape,
                                        auto=False,
                                        scale_up=self.augment)
            shapes = (h0, w0), (
                (h / h0, w / w0), pad)  # for COCO mAP rescaling

            # load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()  # label: class, x, y, w, h
                labels[:, 1] = ratio[0] * w * x[:, 1] + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * x[:, 2] + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3]) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4]) + pad[1]
        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_affine(img,
                                            labels,
                                            degrees=hyp["degrees"],
                                            translate=hyp["translate"],
                                            scale=hyp["scale"],
                                            shear=hyp["shear"])

            # Augment colorspace
            augment_hsv(img,
                        h_gain=hyp["hsv_h"],
                        s_gain=hyp["hsv_s"],
                        v_gain=hyp["hsv_v"])

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xc,yc,w,h
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0-1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True  # 随机水平翻转
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]  # 1 - x_center

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]  # 1 - y_center

        labels_out = torch.zeros((nL, 6))  # nL: number of labels
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return torch.from_numpy(
            img), labels_out, self.img_files[index], shapes, index

    def coco_index(self, index):
        """
        该方法是专门为cocotools统计标签信息准备，最初的代码是yolo格式，
        但此文件改为了coco格式，所以再变换为yolo格式以和后面评估代码兼容
        """
        o_shapes = self.shapes[index][::-1]  # wh to hw

        # load labels
        x = self.labels[index]
        labels = x.copy()  # label: [class, x_min, y_min, width, height]
        # convert to [class, x_center, y_center, width, height]
        labels[:, 1] =  (labels[:, 1] + labels[:, 3] / 2)  # x_center normalized
        labels[:, 2] =  (labels[:, 2] + labels[:, 4] / 2)  # x_center normalized
        # width already normalized
        # height already normalized
        return torch.from_numpy(labels), o_shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, index = zip(*batch)  # transposed
        for i, l in enumerate(label):
            # 就算其中label经过裁切后没有GT框了也没关系
            # 下面batch中的图片id会赋值给一个shape为[0,6]的tensor，但是这个空的tensor在cat时就会消失，所以后面根据image id来计算loss
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, index


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path
        h0, w0 = img.shape[:2]  # orig hw
        # img_size 设置的是预处理后输出的图片尺寸
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[
            index]  # img, hw_original, hw_resized


def load_mosaic(self, index):
    """
    将四张图片拼接在一张马赛克图像中
    :param self:
    :param index: 需要获取的图像索引
    :return:
    """
    # loads images in a mosaic

    labels4 = []  # 拼接图像的label信息
    s = self.img_size
    # 随机初始化拼接图像的中心点坐标
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5))
              for _ in range(2)]  # mosaic center x, y
    # 从dataset中随机寻找三张图像进行拼接
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    # 遍历四张图像进行拼接
    for i, index in enumerate(indices):
        # load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            # 创建马赛克图像
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将截取的图像区域填充到马赛克图像的相应位置
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # 计算pad(图像边界与马赛克边界的距离，越界的情况为负值)
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels 获取对应拼接图像的labels信息
        # [class_index, x_min, y_min, width, height]
        x = self.labels[index]
        labels = x.copy()  # 深拷贝，防止修改原数据
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            # 计算标注数据在马赛克图像中的坐标(绝对坐标变为网络输入尺度)
            labels[:, 1] = w * x[:, 1] + padw  # xmin
            labels[:, 2] = h * x[:, 2] + padh  # ymin
            labels[:, 3] = w * (x[:, 1] + x[:, 3]) + padw  # xmax
            labels[:, 4] = h * (x[:, 2] + x[:, 4]) + padh  # ymax
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # 设置上下限防止越界
        np.clip(labels4[:, 1:], 0, 2 * s,
                out=labels4[:, 1:])  # use with random_affine

    # Augment
    # 随机旋转，缩放，平移以及错切
    img4, labels4 = random_affine(img4,
                                  labels4,
                                  degrees=self.hyp['degrees'],
                                  translate=self.hyp['translate'],
                                  scale=self.hyp['scale'],
                                  shear=self.hyp['shear'],
                                  border=-s // 2)  # border to remove

    return img4, labels4


def random_affine(img,
                  targets=(),
                  degrees=10,
                  translate=.1,
                  scale=.1,
                  shear=10,
                  border=0):
    """随机旋转，缩放，平移以及错切"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # 这里可以参考博文: https://blog.csdn.net/qq_37541097/article/details/119420860
    # targets = [cls, xyxy]

    # 最终输出的图像尺寸，等于img4.shape / 2
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    # 生成旋转以及缩放矩阵
    R = np.eye(3)  # 生成对角阵
    a = random.uniform(-degrees, degrees)  # 随机旋转角度
    s = random.uniform(1 - scale, 1 + scale)  # 随机缩放因子
    R[:2] = cv2.getRotationMatrix2D(angle=a,
                                    center=(img.shape[1] / 2,
                                            img.shape[0] / 2),
                                    scale=s)

    # Translation
    # 生成平移矩阵
    T = np.eye(3)
    T[0,
      2] = random.uniform(-translate, translate
                          ) * img.shape[0] + border  # x translation (pixels)
    T[1,
      2] = random.uniform(-translate, translate
                          ) * img.shape[1] + border  # y translation (pixels)

    # Shear
    # 生成错切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi /
                       180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi /
                       180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        # 进行仿射变化
        img = cv2.warpAffine(img,
                             M[:2],
                             dsize=(width, height),
                             flags=cv2.INTER_LINEAR,
                             borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
            n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        # [4*n, 3] -> [n, 8]
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        # 对transform后的bbox进行修正(假设变换后的bbox变成了菱形，此时要修正成矩形)
        x = xy[:, [0, 2, 4, 6]]  # [n, 4]
        y = xy[:, [1, 3, 5, 7]]  # [n, 4]
        xy = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # [n, 4]

        # reject warped points outside of image
        # 对坐标进行裁剪，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]

        # 计算调整后的每个box的面积
        area = w * h
        # 计算调整前的每个box的面积
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] -
                                                   targets[:, 2])
        # 计算每个box的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    # 这里可以参考博文:https://blog.csdn.net/qq_37541097/article/details/119478023
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat),
                         cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img:
    :param new_shape:
    :param color:
    :param auto:
    :param scale_fill:
    :param scale_up:
    :return:
    """

    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding
    if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding

    img = cv2.copyMakeBorder(img,
                             top,
                             bottom,
                             left,
                             right,
                             cv2.BORDER_CONSTANT,
                             value=color)  # add border
    return img, ratio, (dw, dh)


def create_folder(path="./new_folder"):
    # Create floder
    if os.path.exists(path):
        shutil.rmtree(path)  # dalete output folder
    os.makedirs(path)  # make new output folder


    