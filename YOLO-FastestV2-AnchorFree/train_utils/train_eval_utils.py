import sys

from torch.cuda import amp
import torch.nn.functional as F

from build_utils.utils import *
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
import train_utils.distributed_utils as utils
import wandb

def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq, accumulate, img_size,
                    grid_min, grid_max, gs,
                    multi_scale=False, warmup=False, scaler=None, hyp=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        accumulate = 1
    
    now_lr = 0.
    nb = len(data_loader)  # number of batches
    # imgs: [batch_size, 3, img_size, img_size]
    # targets: [num_obj, 6] , that number 6 means -> (img_index, obj_index, x, y, w, h)
    # paths: list of img path
    for i, (imgs, targets, paths, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # ni 统计从epoch0开始的所有batch数
        ni = i + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        imgs -= torch.tensor([0.485, 0.456, 0.406], dtype=imgs.dtype, device=imgs.device).reshape(1, 3, 1, 1)
        imgs /= torch.tensor([0.229, 0.224, 0.225], dtype=imgs.dtype, device=imgs.device).reshape(1, 3, 1, 1)
        targets = targets.to(device)

        # Multi-Scale
        if multi_scale:
            # 每训练64张图片，就随机修改一次输入图片大小，
            # 由于label已转为相对坐标，故缩放图片不影响label的值
            if ni % accumulate == 0:  # adjust img_size (67% - 150%) every 1 batch
                # 在给定最大最小输入尺寸范围内随机选取一个size(size为32的整数倍)
                img_size = random.randrange(grid_min, grid_max + 1) * gs
            sf = img_size / max(imgs.shape[2:])  # scale factor

            # 如果图片最大边长不等于img_size, 则缩放图片，并将长和宽调整到32的整数倍
            if sf != 1:
                # gs: (pixels) grid size
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)


        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with amp.autocast(enabled=scaler is not None):
            outputs = model(imgs, targets)
        total_loss = outputs["total_loss"]
        iou_loss=outputs["iou_loss"]
        obj_loss=outputs["conf_loss"]
        cls_loss=outputs["cls_loss"]
        num_fg=outputs["num_fg"]
        if not torch.isfinite(total_loss):
            print('WARNING: non-finite loss, ending training ', total_loss)
            print("training image path: {}".format(",".join(paths)))
            sys.exit(1)

        # backward
        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # optimize
        # 每训练64张图片更新一次权重
        if ni % accumulate == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(iou_loss=iou_loss)
        metric_logger.update(obj_loss=obj_loss)
        metric_logger.update(cls_loss=cls_loss)
        metric_logger.update(cls_loss=num_fg)
        metric_logger.update(loss=total_loss)

        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

        if ni % accumulate == 0 and lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

    return iou_loss, obj_loss, cls_loss, total_loss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, coco=None, device=None):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    if coco is None:
        coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for imgs, targets, paths, shapes, img_index in metric_logger.log_every(data_loader, 100, header):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        imgs -= torch.tensor([0.485, 0.456, 0.406], dtype=imgs.dtype, device=imgs.device).reshape(1, 3, 1, 1)
        imgs /= torch.tensor([0.229, 0.224, 0.225], dtype=imgs.dtype, device=imgs.device).reshape(1, 3, 1, 1)
        # targets = targets.to(device)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(imgs)

        output_boxes = non_max_suppression(outputs, conf_thres=0.3, iou_thres=0.3)
        model_time = time.time() - model_time

        outputs = []
        for index, p in enumerate(output_boxes):
            if p is None:
                p = torch.empty((0, 6), device=cpu_device)
                boxes = torch.empty((0, 4), device=cpu_device)
            else:
                # xmin, ymin, xmax, ymax
                boxes = p[:, :4]
                # shapes: (h0, w0), ((h / h0, w / w0), pad)
                # 将boxes信息还原回原图尺度，这样计算的mAP才是准确的
                boxes = scale_coords(imgs[index].shape[1:], boxes, shapes[index][0]).round()

            # 注意这里传入的boxes格式必须是xmin, ymin, xmax, ymax，且为绝对坐标
            info = {"boxes": boxes.to(cpu_device),
                    "labels": p[:, 5].to(device=cpu_device, dtype=torch.int64),
                    "scores": p[:, 4].to(cpu_device)}
            outputs.append(info)

        res = {img_id: output for img_id, output in zip(img_index, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    result_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return result_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
