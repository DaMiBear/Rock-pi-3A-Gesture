import torch.nn as nn
from copy import deepcopy
from thop import profile
from typing import Sequence
import json
import os
import datetime

import torch
from torch.utils import data
import numpy as np
import math
import transforms
from model import Lite_HRNet_SimCC
from my_dataset_coco import CocoKeypoint
from train_utils import train_eval_utils as utils


def create_model(num_joints, pretrain_weights_path=""):
    model = Lite_HRNet_SimCC(num_joints=num_joints, input_size=args.fixed_size,
                          simcc_split_ratio=args.simcc_split_ratio)
    model.init_weights()
    if os.path.exists(pretrain_weights_path):
        weights_dict = torch.load(pretrain_weights_path, map_location='cpu')
        model_dict = model.state_dict()     # 获得当前网络的字典
        weights_dict_name = "state_dict" if "state_dict" in weights_dict else "model"

        for d1, d2 in zip(weights_dict[weights_dict_name].items(), model_dict.items()):
            k = 0
            v = 1
            # 因为修改了网络的类名
            # 只要预训练权重当前层的元素个数一样就载入
            if d1[v].numel() == d2[v].numel():
                model_dict[d2[k]] = d1[v]
            else:
                print("Not match dict key:", d1[k], "<--->", d2[k])

        missing_keys, unexpected_keys = model.load_state_dict(
            model_dict, strict=False)
        if len(missing_keys) != 0:
            print("missing_keys: ", missing_keys)
        print("Use pretrinaed path:", pretrain_weights_path)
    # print(model(torch.ones(((1, 3, 64, 64)))))
    return model


def freeze_model(model):
    param_list = ['mlp_head_x', 'mlp_head_y']
    for k, v in model.named_parameters():
        if not any(x in k.split('.') for x in param_list):
            v.requires_grad_(False)
        else:
            print(f"Did not freeze: {k}")


def get_model_info(model: nn.Module, tsize: Sequence[int], img_channel=3) -> str:
    img = torch.zeros((1, img_channel, tsize[0], tsize[1]),
                      device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= 2  # Gflops
    info = "Params: {:.3f}M, Gflops: {:.3f}".format(params, flops)
    return info


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    # 用来保存coco_info的文件
    results_file = "outputs/results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    with open(args.keypoints_path, "r") as f:
        kps_info = json.load(f)

    fixed_size = args.fixed_size
    # TODO: 暂未分配每个关键点的权重
    # kps_weights = np.array(kps_info["kps_weights"],
    #                        dtype=np.float32).reshape((args.num_joints,))
    data_transform = {
        "train": transforms.Compose([
            # transforms.HalfBody(0.3, kps_info["upper_body_ids"], kps_info["lower_body_ids"]),
            transforms.AffineTransform(
                scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            transforms.RandomHorizontalFlip(0.5, kps_info["flip_pairs"]),
            transforms.KeypointToSimCC(args.coord_representation, simcc_split_ratio=args.simcc_split_ratio,
                                       use_different_keypoints_weight=False, keypoints_weights=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(
                scale=(1.25, 1.25), fixed_size=fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
    }

    data_root = args.data_path

    # load train data set
    # coco2017 -> annotations -> keypoints.json
    train_dataset = CocoKeypoint(
        data_root, "train", transforms=data_transform["train"], fixed_size=args.fixed_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % nw)

    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=False,
                                        num_workers=nw,
                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    # coco2017 -> annotations -> keypoints.json
    val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], fixed_size=args.fixed_size,
                               det_json_path=args.target_det)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=False,
                                      num_workers=nw,
                                      collate_fn=val_dataset.collate_fn)

    # create model
    model = create_model(num_joints=args.num_joints,
                         pretrain_weights_path=args.pretrained_path)
    # print(model)
    print(get_model_info(model, args.fixed_size, 3))
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    def lf(x): 
        return ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler, args=args)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()
        if epoch % args.interval == 0 or (epoch == args.epochs - 1):
            # evaluate on the test dataset
            coco_info = utils.evaluate(model, val_data_loader, device=device, simcc_split_ratio=args.simcc_split_ratio,
                                       flip=False, flip_pairs=kps_info["flip_pairs"])

            # write into txt
            with open(results_file, "a") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [f"{i:.4f}" for i in coco_info +
                               [mean_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            val_map.append(coco_info[1])  # @0.5 mAP

            # save weights
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            torch.save(save_files, "./save_weights/model-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument(
        '--device', default='cuda:0', help='device')
    # 训练数据集的根目录(coco格式)
    parser.add_argument(
        '--data-path', default='', help='dataset')
    # COCO数据集关键点信息
    parser.add_argument(
        '--keypoints-path', default="hand_21_keypoints.json", type=str,
        help='keypoints.json path')
    # 预训练权重来初始化网络
    parser.add_argument(
        '--pretrained-path', default=".pth", type=str,
        help='pretrained weights to init net')
    # sa-SimCC Or SimCC
    parser.add_argument(
        '--coord-representation', default="simcc", type=str,
        help='sa-simcc Or simcc')
    # 原项目提供的验证集检测信息，如果要使用GT信息，直接将该参数置为None，建议设置成None
    parser.add_argument(
        '--target-det', type=str, default=None)
    parser.add_argument(
        '--fixed-size', default=[128, 128], nargs='+', type=int, help='input size')
    parser.add_argument(
        '--simcc-split-ratio', default=2.0,
        type=float, help='simcc split ratio')

    # keypoints点数
    parser.add_argument(
        '--num-joints', default=21,
        type=int, help='num_joints')
    # 评估间隔
    parser.add_argument(
        '--interval', default=5,
        type=int, help='eval interval')
    # 文件保存地址
    parser.add_argument(
        '--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument(
        '--resume', default='', type=str,
        help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument(
        '--start-epoch', default=0,
        type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument(
        '--epochs', default=100, type=int, metavar='N',
        help='number of total epochs to run')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument(
        '--lr-steps', default=[180, 200], nargs='+', type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument(
        '--lr-gamma', default=0.1, type=float,
        help='decrease lr by a factor of lr-gamma')
    # 学习率
    parser.add_argument(
        '--lr', default=1e-3, type=float,
        help='initial learning rate, 0.02 is the default value for training '
        'on 8 gpus and 2 images_per_gpu')
    # AdamW的weight_decay参数
    parser.add_argument(
        '--wd', '--weight-decay', default=1e-4, type=float,
        metavar='W', help='weight decay (default: 1e-4)',
        dest='weight_decay')
    # 训练的batch size
    parser.add_argument(
        '--batch-size', default=64, type=int, metavar='N',
        help='batch size when training.')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument(
        "--amp", default=True, action="store_true",
        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
