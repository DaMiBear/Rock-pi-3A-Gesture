import torch
import torch.nn as nn
import build_utils

from models.fpn import *
from models.backbone.shufflenetv2 import *
from build_utils.utils import bboxes_iou, IOUloss, FocalLoss

class Detector(nn.Module):
    def __init__(self, classes, anchor_num, init_backbone, use_focal_loss=False, export_onnx=False, hyp=None):
        super(Detector, self).__init__()
        out_depth = 72
        stage_out_channels = [-1, 24, 48, 96, 192]

        self.strides = [16, 32]
        self.grids = [torch.zeros(1)] * len(self.strides)  # 只有两个尺度
        self.num_classes = classes
        self.n_anchors = anchor_num

        self.export_onnx = export_onnx
        self.backbone = ShuffleNetV2(stage_out_channels, init_backbone)
        self.fpn = LightFPN(stage_out_channels[-2] + stage_out_channels[-1], stage_out_channels[-1], out_depth)

        self.output_reg_layers = nn.Conv2d(out_depth, 4 * anchor_num, 1, 1, 0, bias=True)
        self.output_obj_layers = nn.Conv2d(out_depth, anchor_num, 1, 1, 0, bias=True)
        self.output_cls_layers = nn.Conv2d(out_depth, classes, 1, 1, 0, bias=True)

        if (use_focal_loss):
            self.bcewithlog_loss = FocalLoss(nn.BCEWithLogitsLoss(reduction="none"), gamma=hyp["focal_gamma"], alpha=hyp["focal_alpha"])
        else:
            self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        
        self.iou_loss = IOUloss(reduction="none")

    def forward(self, x, labels=None):
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        out_reg = []
        out_obj = []
        out_cls = []
        outputs = []

        C2, C3 = self.backbone(x)
        cls_2, obj_2, reg_2, cls_3, obj_3, reg_3 = self.fpn(C2, C3)

        out_reg.append(self.output_reg_layers(reg_2))
        out_obj.append(self.output_obj_layers(obj_2))
        out_cls.append(self.output_cls_layers(cls_2))

        out_reg.append(self.output_reg_layers(reg_3))
        out_obj.append(self.output_obj_layers(obj_3))
        out_cls.append(self.output_cls_layers(cls_3))


        for k, (reg_output, obj_output, cls_output) in enumerate(zip(out_reg, out_obj, out_cls)):
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                # 输入变为网络输入大小下的xc yc w h 并且产生当前尺度的grid
                assert x.shape[2] / reg_output.shape[2] == x.shape[3] / reg_output.shape[3], "特征图宽高下采样不一致"
                stride_this_level = x.shape[2] / reg_output.shape[2]
                output, grid = self.get_output_and_grid(output, k, stride_this_level, x.type())  # 这里output会变成输入尺寸xc yc w h
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(x)
                )
            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

            outputs.append(output)


        if self.training:
            assert labels is not None
            loss, iou_loss, conf_loss, cls_loss, num_fg = self.get_losses(x,
                                                                          x_shifts,
                                                                          y_shifts,
                                                                          expanded_strides,
                                                                          labels,
                                                                          torch.cat(outputs, 1),
                                                                          dtype=x.dtype)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            return outputs
        else:
            if self.export_onnx:
                # 把(4+na+nc)移动到最后一维，方便读取
                # outputs = [out.permute(0, 2, 3, 1) for out in outputs]
                return [out.permute(0, 2, 3, 1) for out in outputs]
            else:
                self.hw = [x.shape[-2:] for x in outputs]  # 记录每个尺度head特征图的hw
                # [batch, n_anchors_all, 4+1+num_classes]
                outputs = torch.cat(
                    [x.flatten(start_dim=2) for x in outputs], dim=2
                ).permute(0, 2, 1)
                return self.decode_outputs(outputs, dtype=x.type())


    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid
    
        
    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, dtype):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets  labels[n_label, 6]  6:batch中图片id、类别、xc、yc、w、h
        batch_size = outputs.shape[0]
        nlabel = torch.zeros(batch_size, dtype=dtype, device=labels.device)  # number of objects
        for i in range(batch_size):
            nlabel[i] = (labels[:, 0] == i).sum()

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)  # [1, n_anchors_all]
     
        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                # 如果图片中没有GT框，这个图片就没有正样本，fg_mask全为False，后面计算iou loss
                # 和cls loss时就会被过滤掉 obj loss就全没有目标
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                labels_mask_per_image = labels[:, 0] == batch_idx
                gt_bboxes_per_image = labels[labels_mask_per_image, 2:6]
                _, _, imgs_h, imgs_w = imgs.shape
                # 变为网络输入大小下的坐标表示
                gt_bboxes_per_image[:, 0::2] = gt_bboxes_per_image[:, 0::2] * imgs_w
                gt_bboxes_per_image[:, 1::2] = gt_bboxes_per_image[:, 1::2] * imgs_h
                gt_classes = labels[labels_mask_per_image, 1]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(  # noqa 初筛以及精选正样本
                    batch_idx,
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                    cls_preds,
                    bbox_preds,
                    obj_preds,
                    labels,
                    imgs,
                )
             
                torch.cuda.empty_cache()
                num_fg += num_fg_img  # 统计每张图片正样本个数

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)  # 正样本的类别索引变为onehot后*对应iou [num_fg, num_classes]
                obj_target = fg_mask.unsqueeze(-1)  # 正负样本 [total_anchors, 1]
                reg_target = gt_bboxes_per_image[matched_gt_inds]  # 正样本GT框[num_fg, 4]
            # 记录每张图片的正样本对应的GT信息
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)   # fg_mask shape[total_anchors, ]
        

        cls_targets = torch.cat(cls_targets, 0)  # [total_num_fg, num_classes]
        reg_targets = torch.cat(reg_targets, 0)  # [total_num_fg, 4]
        obj_targets = torch.cat(obj_targets, 0)  # [total_anchors*batch_size, 1]
        fg_masks = torch.cat(fg_masks, 0)  # [total_anchors*batch_size, ]


        num_fg = max(num_fg, 1)  # 防止一个batch中都没有正样本？
        # 计算loss 预测框和按照索引选取的GT正样本
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            num_fg / max(num_gts, 1),  # 一个batch中正样本个数/一个batch中GT个数
        )

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # 初筛正样本 fg_mask[total_anchor, ]初筛的正样本mask，is_in_boxes_and_center[n_gt, 初筛正样本个数]同时在GT和center的mask
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )
        # 当前图片中初筛正样本对应的预测
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (  # [n_gt, num_in_boxes_anchor, num_classes]
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)  # [n_gt, num_in_boxes_anchor]

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(self,
                          gt_bboxes_per_image,
                          expanded_strides,
                          x_shifts,
                          y_shifts,
                          total_num_anchors,
                          num_gt):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # [num_gt, total_anchors] 找出中心点在GT框内的锚框
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0  # [total_anchors, ] >0表示这个索引对应的锚框可以为正样本
        # in fixed center

        center_radius = 1.5
        # 理解为对center_x center_y 分别左上平移到一个点 右下平移到一个点。来产生一个区域
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0  # [num_gt, total_anchor]
        is_in_centers_all = is_in_centers.sum(dim=0) > 0  # [total_anchor, ]

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()  # [num_gt,] 给每个GT框分配多少个预测框
        for gt_idx in range(num_gt):  # 找出与GT框的损失最小的前k个预测框作为正样本
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )  # cost最小的dynamic_ks[gt_idx]个的索引
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:  # >1表示存在预测框被分配给多个GT，则只分配给cost最小的那个
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0  # 精筛之后的预测框索引mask
        num_fg = fg_mask_inboxes.sum().item()  # 一共正样本预测框个数

        fg_mask[fg_mask.clone()] = fg_mask_inboxes  # 在原来初筛的位置上 用精筛的覆盖掉

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]  # 返回值：正样本锚框(预测框)个数、每个正样本的类别、GT和正样本的IOU、正样本对应的GT索引
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides  # 网络输出的是中心偏移，实际预测框需要加上格点坐标再乘步长
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides  # wh求exp在乘步长
        return outputs
