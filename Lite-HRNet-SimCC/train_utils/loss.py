import torch
import torch.nn as nn


class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # [num_kps] -> [B, num_kps]
        kps_weights = torch.stack(
            [t["kps_weights"].to(device) for t in targets])

        # [B, num_kps, H, W] -> [B, num_kps]
        # loss = self.criterion(logits, heatmaps).mean(dim=[0,2, 3])
        # loss = torch.sum(loss * kps_weights) / bs

        # [B, num_kps, H, W] -> [1,] 暂时去除了权重
        loss = self.criterion(logits, heatmaps).mean()
        return loss


class NMTCritierion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]

        if label_smoothing > 0:
            self.criterion_ = nn.KLDivLoss(reduction='none')
        else:
            self.criterion_ = nn.NLLLoss(reduction='none', ignore_index=100000)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)

        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach().type(torch.int64)
            # Do label smoothing, shape is [M]
            one_hot = self._smooth_label(num_tokens)
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            # after tdata.unsqueeze(1) , tdata shape is [N,1]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            gtruth = tmp_.detach()
        loss = torch.sum(self.criterion_(scores, gtruth), dim=1)
        return loss

    def forward(self, output_x, output_y, targets):
        # output_x shape:[bs, nkpt, image_w * ratio]
        # output_t shape:[bs, nkpt, image_h * ratio]
        device = output_x.device
        batch_size = output_x.size(0)
        num_kpts = output_x.size(1)
        gt_keypoints = torch.stack(
            [t["keypoints"].to(device) for t in targets])   # shape:[bs, nkpt, 2]
        gt_keypoints_weights = torch.stack(
            [t["kps_weights"].to(device) for t in targets])     # shape:[bs, nkpt]
        loss = 0

        for idx in range(num_kpts):
            # shape:[bs, image_w * ratio]
            coord_x_pred = output_x[:, idx].squeeze()
            coord_y_pred = output_y[:, idx].squeeze()
            coord_gt = gt_keypoints[:, idx].squeeze()    # shape:[bs, 2]
            weight = gt_keypoints_weights[:, idx].squeeze()      # shape:[bs,]
            loss += self.criterion(coord_x_pred,
                                   coord_gt[:, 0]).mul(weight).sum()
            loss += self.criterion(coord_y_pred,
                                   coord_gt[:, 1]).mul(weight).sum()
        return loss / batch_size


class KLDiscretLoss(nn.Module):
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1) #[B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='none')
 
 
    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim=1) 
        return loss

    def forward(self, output_x, output_y, targets):
        device = output_x.device
        num_joints = output_x.size(1)
        loss = 0

        target_x = torch.stack(
            [t["keypoints_x"].to(device) for t in targets])   # shape:[bs, nkpt, image_w * ratio]
        target_y = torch.stack(
            [t["keypoints_y"].to(device) for t in targets])   # shape:[bs, nkpt, image_h * ratio]
        target_weight = torch.stack(
            [t["kps_weights"].to(device) for t in targets])     # shape:[bs, nkpt]
        for idx in range(num_joints):
            coord_x_pred = output_x[:,idx].squeeze()
            coord_y_pred = output_y[:,idx].squeeze()
            coord_x_gt = target_x[:,idx].squeeze()
            coord_y_gt = target_y[:,idx].squeeze()
            weight = target_weight[:,idx].squeeze()
            loss += (self.criterion(coord_x_pred,coord_x_gt).mul(weight).mean()) 
            loss += (self.criterion(coord_y_pred,coord_y_gt).mul(weight).mean())
        return loss / num_joints 
