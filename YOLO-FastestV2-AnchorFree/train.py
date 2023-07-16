import datetime
import argparse

import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from build_utils.datasets import *
from build_utils.datasets_coco import LoadImagesAndLabels_COCO
from build_utils.parse_config import parse_data_cfg
from build_utils.utils import *
from train_utils import train_eval_utils as train_util
from train_utils import get_coco_api_from_dataset

import models.detector
import wandb


from copy import deepcopy
from thop import profile
from typing import Sequence
def get_model_info(model: nn.Module, tsize: Sequence[int], img_channel=3) -> str:
    img = torch.zeros((1, img_channel, tsize[0], tsize[1]),
                      device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= 2  # Gflops
    info = "Params: {:.3f}M, Gflops: {:.3f}".format(params, flops)
    return info


def train(hyp):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    wdir = "weights" + os.sep  # weights dir
    best = wdir + "best.pt"
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    imgsz_train = opt.img_size
    imgsz_test = opt.img_size  # test image sizes
    multi_scale = opt.multi_scale

    # Image sizes
    # 图像要设置成32的倍数
    gs = 32  # (pixels) grid size
    assert math.fmod(imgsz_test, gs) == 0, "--img-size %g must be a %g-multiple" % (imgsz_test, gs)
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs
    if multi_scale:
        imgsz_min = opt.img_size // 1.5
        imgsz_max = opt.img_size // 0.667

        # 将给定的最大，最小输入尺寸向下调整到32的整数倍
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max  # initialize with max size
        print("Using multi_scale training, image range[{}, {}]".format(imgsz_min, imgsz_max))

    # configure run
    data_dict = parse_data_cfg(data)
    # 类型转换
    data_dict["classes"] = int(data_dict["classes"])
    data_dict["anchor_num"] = int(data_dict["anchor_num"])

    nc = 1 if opt.single_cls else int(data_dict["classes"])  # number of classes

    # Remove previous results
    for f in glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = models.detector.Detector(data_dict["classes"], data_dict["anchor_num"], 
                                    init_backbone=True, 
                                    use_focal_loss=opt.use_focal_loss, 
                                    hyp=hyp).to(device)
    print(get_model_info(model=model, tsize=(imgsz_test, imgsz_test), img_channel=3))
    if os.path.exists(opt.weights):
        model.load_state_dict(torch.load(opt.weights, map_location=device)["model"], strict=False)
        print("Load finetune model param: %s" % opt.weights)

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=hyp["lr0"], momentum=hyp["momentum"],
                          weight_decay=hyp["weight_decay"], nesterov=True)

    scaler = torch.cuda.amp.GradScaler() if opt.amp else None

    start_epoch = 0
    best_map = 0.0

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  # 指定从哪个epoch开始

    """ YOLO格式 """
    # dataset
    # 训练集的图像尺寸指定为multi_scale_range中最大的尺寸
    # train_dataset = LoadImagesAndLabels(train_path, imgsz_train, batch_size,
    #                                     augment=True,
    #                                     hyp=hyp,  # augmentation hyperparameters
    #                                     rect=opt.rect,  # rectangular training
    #                                     cache_images=opt.cache_images,
    #                                     single_cls=opt.single_cls)
    # 验证集的图像尺寸指定为img_size(352)
    # val_dataset = LoadImagesAndLabels(test_path, imgsz_test, batch_size,
    #                                   hyp=hyp,
    #                                   rect=True,  # 将每个batch的图像调整到合适大小，可减少运算量(并不是352x352标准尺寸)
    #                                   cache_images=opt.cache_images,
    #                                   single_cls=opt.single_cls)
    """ COCO格式 """
    train_dataset = LoadImagesAndLabels_COCO(path=data_dict["dataset_dir"],
                                             images_dir=data_dict["train_images_dir"],
                                             json_file=data_dict["train_json_file"],
                                             img_size=imgsz_train,
                                             batch_size=batch_size, augment=True, hyp=hyp, 
                                             rect=opt.rect,
                                             cache_images=opt.cache_images,
                                             single_cls=opt.single_cls)
    
    val_dataset = LoadImagesAndLabels_COCO(path=data_dict["dataset_dir"],
                                           images_dir=data_dict["val_images_dir"],
                                           json_file=data_dict["val_json_file"],
                                           img_size=imgsz_test,
                                           batch_size=batch_size, augment=False, hyp=hyp, 
                                           rect=True,
                                           cache_images=opt.cache_images,
                                           single_cls=opt.single_cls)

    # dataloader
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=not opt.rect,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

    val_datasetloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=nw,
                                                    pin_memory=True,
                                                    collate_fn=val_dataset.collate_fn)

    # start training
    # caching val_data when you have plenty of memory(RAM)
    # coco = None
    coco = get_coco_api_from_dataset(val_dataset)

    print("starting traning for %g epochs..." % epochs)
    print('Using %g dataloader workers' % nw)
    print("Hyp:", hyp)
    for epoch in range(start_epoch, epochs):
        if epoch >= epochs - hyp["stop_mosaic_epoch"]:
            train_dataloader.dataset.stop_mosaic = True

        iou_loss, obj_loss, cls_loss, total_loss, lr = train_util.train_one_epoch(model, optimizer, train_dataloader,
                                               device, epoch,
                                               accumulate=accumulate,  # 迭代多少batch才训练完64张图片
                                               img_size=imgsz_train,  # 输入图像的大小
                                               multi_scale=multi_scale,
                                               grid_min=grid_min,  # grid的最小尺寸
                                               grid_max=grid_max,  # grid的最大尺寸
                                               gs=gs,  # grid step: 32
                                               print_freq=50,  # 每训练多少个step打印一次信息
                                               warmup=True,
                                               scaler=scaler,
                                               hyp=hyp)
        # update scheduler
        scheduler.step()

        if opt.notest is False or epoch == epochs - 1:
            # evaluate on the test dataset
            result_info = train_util.evaluate(model, val_datasetloader,
                                              coco=coco, device=device)
            coco_mAP = result_info[0]
            voc_mAP = result_info[1]
            coco_mAR = result_info[8]
            wandb.log({
                "iou_loss": iou_loss,
                "obj_loss": obj_loss,
                "cls_loss": cls_loss,
                "total_loss": total_loss,
                "learning_rate": lr,
                "mAP@0.5:0.95": coco_mAP,
                "mAP@0.5": voc_mAP,
                "mAR@0.5:0.95": coco_mAR
            })
            # write into txt
            with open(results_file, "a") as f:
                # 记录coco的12个指标加上训练总损失和lr
                result_info = [str(round(i, 4)) for i in result_info + [total_loss.tolist()]] + [str(round(lr, 6))]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            # update best mAP(IoU=0.50:0.95)
            if coco_mAP > best_map:
                best_map = coco_mAP

            if opt.savebest is False:
                # save weights every epoch
                with open(results_file, 'r') as f:
                    save_files = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'training_results': f.read(),
                        'epoch': epoch,
                        'best_map': best_map}
                    if opt.amp:
                        save_files["scaler"] = scaler.state_dict()
                    torch.save(save_files, "./weights/YOLO-FastestV2-AnchorFree-{}.pt".format(epoch))
            else:
                # only save best weights
                if best_map == coco_mAP:
                    with open(results_file, 'r') as f:
                        save_files = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map}
                        if opt.amp:
                            save_files["scaler"] = scaler.state_dict()
                        torch.save(save_files, best.format())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--use-focal-loss', type=bool, default=False)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data', type=str, default='data/mydata.data', help='*.data path')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--multi-scale', type=bool, default=True,
                        help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=352, help='test size')
    parser.add_argument('--rect', default=False, action='store_true', help='rectangular training')
    parser.add_argument('--savebest', type=bool, default=True, help='only save best checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    opt = parser.parse_args()

    # 检查文件是否存在
    opt.data = check_file(opt.data)
    opt.hyp = check_file(opt.hyp)
    print(opt)

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login()
    wandb.init(project="YOLO-FastestV2-AnchorFree")     # 改成自己的 https://docs.wandb.ai/quickstart
    wandb.config = {
        "epochs": opt.epochs,
        "batch_size": opt.batch_size
    }
    train(hyp)
