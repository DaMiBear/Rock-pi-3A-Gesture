import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import channel_shuffle, normal_init, constant_init, linear_init
from torch.nn.modules.batchnorm import _BatchNorm


BN_MOMENTUM = 0.1

class ConvModule(nn.Module):
    """
        Conv2d -> BN -> [activate]
    """
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=2, padding=1, groups=1, activate='ReLU'):
        super().__init__()
        assert activate in ['ReLU', 'Sigmoid', None], 'activate must be ReLU or Sigmoid or None'
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.activate = None
        if activate == 'ReLU':
            self.activate = nn.ReLU(inplace=True)
        elif activate == 'Sigmoid':
            self.activate = nn.Sigmoid()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.activate is not None:
            out = self.activate(out)
        return out

class Stem(nn.Module):
    def __init__(self, in_channels, stem_channels, out_channels, expand_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvModule(
            in_channels=in_channels, 
            out_channels=stem_channels, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            activate='ReLU'
            )
        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            ConvModule(
                in_channels=branch_channels,
                out_channels=branch_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=branch_channels,
                activate=None),
            ConvModule(
                in_channels=branch_channels,
                out_channels=inc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activate='ReLU'),
        )

        self.expand_conv = ConvModule(
            branch_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activate='ReLU')
        self.depthwise_conv = ConvModule(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=mid_channels,
            activate=None)
        self.linear_conv = ConvModule(
            mid_channels,
            branch_channels
            if stem_channels == self.out_channels else stem_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activate='ReLU')

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)

        x2 = self.expand_conv(x2)
        x2 = self.depthwise_conv(x2)
        x2 = self.linear_conv(x2)

        out = torch.cat((self.branch1(x1), x2), dim=1)

        out = channel_shuffle(out, 2)
        return out


class DepthwiseSeparableConvModule(nn.Module):
    """
        depthwise conv2d, bn, pointwise conv2d, bn, relu
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # depthwise conv2d, bn
        self.dw_convs = ConvModule(in_channels, in_channels, kernel_size=3, stride=stride, 
                                    padding=1, groups=in_channels, activate=None)
        # pointwise conv2d, bn, relu
        self.pw_convs = ConvModule(in_channels, out_channels, kernel_size=1, stride=1, 
                                    padding=0, activate='ReLU')
        
    def forward(self, x):
        out = self.dw_convs(x)
        out = self.pw_convs(out)

        return out

class CrossResolutionWeighting(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()
        self.channels = channels
        # self.pool2d_kernel_size = [8, 4, 2]
        total_channel = sum(channels)
        # 因为ONNX不支持自适应池化，所以这里计算好固定的池化层kernel大小
        # 全部池化到最小的大小
        # self.avg_pool_list = nn.ModuleList([
        #     nn.AvgPool2d(k) for k in self.pool2d_kernel_size[-(len(self.channels) - 1):]
        # ])
        self.conv1 = ConvModule(in_channels=total_channel,
                                out_channels=int(total_channel / ratio),
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                activate='ReLU')
        self.conv2 = ConvModule(in_channels=int(total_channel / ratio),
                                out_channels=total_channel,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                activate='Sigmoid')
    
    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        # out = [pool2d(s) for pool2d, s in zip(self.avg_pool_list, x[:-1])] + [x[-1]]    # 全部池化到x[-1]的shape
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)

        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
            for s, a in zip(x, out)
        ]
        return out

    
class SpatialWeighting(nn.Module):
    def __init__(self, channels, ratio=4):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=int(channels / ratio),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(channels / ratio),
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 输入x是一个tensor，不是list
        out = self.global_avgpool(x)
        # out = x.mean(dim=(2, 3), keepdim=True)  # 用均值来代替全局平均池化
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class ConditionalChannelWeighting(nn.Module):
    def __init__(self, in_channels, stride, reduce_ratio):
        super().__init__()
        assert stride in [1,2]
        self.stride = stride
        # 会在通道维度把tensor分成两半
        branch_channels = [channel // 2 for channel in in_channels]
        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels,
            ratio=reduce_ratio)
        
        self.depthwise_convs = nn.ModuleList([
            ConvModule(
                channel,
                channel,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=channel,
                activate=None) for channel in branch_channels
        ])

        self.spatial_weighting = nn.ModuleList([
            SpatialWeighting(channels=channel, ratio=4)
            for channel in branch_channels
        ])

    def forward(self, x):
        # 输入x是一个list
        x = [s.chunk(2, dim=1) for s in x]
        x1 = [s[0] for s in x]
        x2 = [s[1] for s in x]

        x2 = self.cross_resolution_weighting(x2)
        x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
        x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

        out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
        out = [channel_shuffle(s, 2) for s in out]

        return out


class LiteHRModule(nn.Module):
    def __init__(self, num_input_branches, num_output_branches, 
                 num_blocks, in_channels, reduce_ratio, stride=1):
        """
            构建Stage
            num_input_branches: 输入分支数
            num_output_branches: 输出分支数
            num_blocks: block个数
            in_channels: 每个尺度的通道数
            ConditionalChannelWeighting x num_blocks + fuse_layer
        """
        super().__init__()
        self.num_input_branches = num_input_branches
        self.num_output_branches = num_output_branches
        self.in_channels = in_channels
        layers = []
        for _ in range(num_blocks):
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio
                )
            )
        self.block_layers = nn.Sequential(*layers)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.num_output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.num_input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif j > i:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(self.in_channels[j], self.in_channels[i], kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(self.in_channels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # j < i
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    conv_downsamples = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for _ in range(i - j - 1):
                        conv_downsamples.append(
                            # depthwise conv2d, bn, pointwise conv2d, bn, relu
                            DepthwiseSeparableConvModule(
                                in_channels=self.in_channels[j],
                                out_channels=self.in_channels[j],
                                stride=2)
                        )
                    # k == i - j - 1最后一个卷积层不仅要调整通道，还要进行下采样
                    conv_downsamples.append(
                        nn.Sequential(
                            # DW, BN
                            ConvModule(
                                in_channels=self.in_channels[j],
                                out_channels=self.in_channels[j],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=self.in_channels[j],
                                activate=None),
                            # PW, BN通道改变
                            ConvModule(
                                in_channels=self.in_channels[j],
                                out_channels=self.in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                activate=None)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*conv_downsamples))

        self.relu = nn.ReLU()   # 注意这里inplace=False

    def forward(self, x):
        # x是List，先通过几个block
        out = self.block_layers(x)
        # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)):
            # 用block最低通道的输出来初始化y
            # `y = 0` will lead to decreased accuracy (0.5~1 mAP)

            y = self.fuse_layers[i][0](out[0])
            for j in range(self.num_input_branches):
                y += self.fuse_layers[i][j](out[j])
            x_fused.append(self.relu(y))

            # y = 0
            # for j in range(self.num_input_branches):
            #     if j == 0:
            #         y = 2.0 * self.fuse_layers[i][j](out[j])
            #     else:
            #         y += self.fuse_layers[i][j](out[j])
            # x_fused.append(self.relu(y))

            # x_fused.append(
            #     self.relu(
            #         sum([
            #             self.fuse_layers[i][j](out[j]) for j in range(self.num_input_branches)]
            #         )
            #     )
            # )

        return x_fused


class IterativeHead(nn.Module):
    """
    Extra iterative head for feature learning.

    Args:
        in_channels (int): The input channels of the block.
    """
    def __init__(self, in_channels):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]    # 从最大的通道开始

        for i in range(num_branchs):
            if i != num_branchs - 1:
                # 前几层逐渐降低通道数
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i + 1],
                        stride=1
                    ))
            else:
                # 最后一层通道数不变，为最低通道数
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i],
                        stride=1
                    ))
        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]     # 从最大的通道数开始

        y = []
        last_x = None
        # 从最大的通道层的输出，每次卷积减少通道数
        # 卷积(通道数减少) -> 上采样(高宽翻倍) -> 求和
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x,
                    size=s.size()[-2:],
                    mode='bilinear',
                    align_corners=True
                )
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s

        return y[::-1]


class Lite_HRNet_SimCC(nn.Module):
    def __init__(self, num_joints: int, input_size: list, simcc_split_ratio: float, export_ncnn=False):
        """
        num_joints: 关键点个数
        input_size: 网络输入图像大小h, w
        simcc_split_ratio: simcc把输入图像扩张的比例
        """
        super().__init__()
        # 必须是2倍关系
        # num_channels = [
        #     [40, 80],
        #     [40, 80, 160],
        #     [40, 80, 160, 320],
        # ]
        num_channels = [
            [24, 48],
            [24, 48, 96],
            [24, 48, 96, 192],
        ]
        self.num_joints = num_joints
        self.export_ncnn = export_ncnn
        # Stem
        stem_channels = 32
        self.stem = Stem(3, stem_channels=stem_channels, out_channels=32, expand_ratio=1)

        # transition1 
        self.transition1 = nn.ModuleList([
            DepthwiseSeparableConvModule(stem_channels, num_channels[0][0], stride=1),
            DepthwiseSeparableConvModule(stem_channels, num_channels[0][1], stride=2),
        ])

        # Stage2, channels:30, 60
        self.stage2 = nn.Sequential(
            LiteHRModule(num_input_branches=2, num_output_branches=2, num_blocks=2, 
                        in_channels=num_channels[0], reduce_ratio=8, stride=1),
            LiteHRModule(num_input_branches=2, num_output_branches=2, num_blocks=2, 
                        in_channels=num_channels[0], reduce_ratio=8, stride=1)
        )

        # transition2
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            DepthwiseSeparableConvModule(
                in_channels=num_channels[0][-1],
                out_channels=num_channels[1][-1],
                stride=2
            )
        ])

        # Stage3 channels: 30, 60, 120
        self.stage3 = nn.Sequential(
            LiteHRModule(num_input_branches=3, num_output_branches=3, num_blocks=2,
                        in_channels=num_channels[1], reduce_ratio=8, stride=1),
            LiteHRModule(num_input_branches=3, num_output_branches=3, num_blocks=2,
                        in_channels=num_channels[1], reduce_ratio=8, stride=1),
            LiteHRModule(num_input_branches=3, num_output_branches=3, num_blocks=2,
                        in_channels=num_channels[1], reduce_ratio=8, stride=1),
            LiteHRModule(num_input_branches=3, num_output_branches=3, num_blocks=2,
                        in_channels=num_channels[1], reduce_ratio=8, stride=1),
        )

        # transition3
        self.transition3 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            DepthwiseSeparableConvModule(
                in_channels=num_channels[1][-1],
                out_channels=num_channels[2][-1],
                stride=2
            )
        ])

        # Stage4 channels: 30, 60, 120, 240
        self.stage4 = nn.Sequential(
            LiteHRModule(num_input_branches=4, num_output_branches=4, num_blocks=2,
            in_channels=num_channels[2], reduce_ratio=8, stride=1),
            LiteHRModule(num_input_branches=4, num_output_branches=4, num_blocks=2,
            in_channels=num_channels[2], reduce_ratio=8, stride=1),
            # LiteHRModule(num_input_branches=4, num_output_branches=1, num_blocks=2,
            # in_channels=num_channels[2], reduce_ratio=8, stride=1),

        )

        # Final layer
        self.final_layer = IterativeHead(in_channels=num_channels[-1])

        # keypoint head
        self.keypoint_head = nn.Conv2d(
            in_channels=num_channels[0][0],
            out_channels=self.num_joints,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        # extra SimCC head
        self.mlp_head_x = nn.Linear(int((input_size[0] // 4) * (input_size[1] // 4)), 
                                    int(input_size[1] * simcc_split_ratio))
        self.mlp_head_y = nn.Linear(int((input_size[0] // 4) * (input_size[1] // 4)), 
                                    int(input_size[0] * simcc_split_ratio))

    def forward(self, x):
        # stem
        x = self.stem(x)
        # transition1
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list

        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        x = self.final_layer(x)
        
        x = self.keypoint_head(x[0]) # after head, x is a tensor

        # extra SimCC head
        x = torch.flatten(x, start_dim=2)  # [b,c,h,w] -> [b,c,h*w]
        pred_x = self.mlp_head_x(x)     # shape:[b, input_w * simcc_split_ratio]
        pred_y = self.mlp_head_y(x)     # shape:[b, input_h * simcc_split_ratio]
        if self.export_ncnn:
            return nn.functional.softmax(pred_x, dim=2), nn.functional.softmax(pred_y, dim=2), 
        else:
            return pred_x, pred_y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                linear_init(m, std=0.001)
                

       
class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        reduction = 'none' if use_target_weight else 'mean'
        self.criterion = nn.MSELoss(reduction=reduction)
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss_joint = self.criterion(heatmap_pred, heatmap_gt)
                loss_joint = loss_joint * target_weight[:, idx]
                loss += loss_joint.mean()
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints * self.loss_weight


    