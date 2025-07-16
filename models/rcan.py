import math
from argparse import Namespace

import jittor as jt
import jittor.nn as nn

from models import register


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Module):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__()
        std = jt.array(rgb_std)
        self.weight = jt.eye(3).view(3, 3, 1, 1)
        self.weight = self.weight / std.view(3, 1, 1, 1)
        self.bias = sign * rgb_range * jt.array(rgb_mean)
        self.bias = self.bias / std
        self.weight.stop_grad()  # 冻结参数，不参与训练
        self.bias.stop_grad()
        self.requires_grad = False  # 整体禁用梯度计算

    def execute(self, x):
        # 模拟 Conv2d 操作：权重卷积 + 偏置
        return nn.conv2d(x, self.weight, self.bias)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # 检查是否为 2 的幂次
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))  # 像素重排，放大 2 倍
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())  # 激活函数（如 ReLU）
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))  # 放大 3 倍
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError(f"不支持的缩放倍数: {scale}")

        super(Upsampler, self).__init__(*m)

## 通道注意力（CA）层
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # 全局平均池化：将特征图压缩为单个点
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 特征通道降维与升维 -> 通道权重
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(),  # Jittor 不支持 inplace 参数，移除
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def execute(self, x):
        y = self.avg_pool(x)  # 全局池化
        y = self.conv_du(y)   # 计算通道权重
        return x * y  # 权重与原特征相乘

## 残差通道注意力块（RCAB）
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(), res_scale=1):  # 移除 inplace 参数

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)  # 第一个卷积后加激活
        modules_body.append(CALayer(n_feat, reduction))  # 通道注意力层
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def execute(self, x):
        res = self.body(x)
        # res = self.body(x) * self.res_scale  # 可选：缩放残差
        res += x  # 残差连接
        return res

## 残差组（RG）
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, 
                bias=True, bn=False, act=act, res_scale=1) 
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))  # 最后一个卷积
        self.body = nn.Sequential(*modules_body)

    def execute(self, x):
        res = self.body(x)
        res += x  # 组内残差连接
        return res

## 残差通道注意力网络（RCAN）
class RCAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RCAN, self).__init__()
        self.args = args

        n_resgroups = args.n_resgroups  # 残差组数量
        n_resblocks = args.n_resblocks  # 每个残差组中的 RCAB 数量
        n_feats = args.n_feats          # 特征通道数
        kernel_size = 3
        reduction = args.reduction      # 通道注意力的降维率
        scale = args.scale[0]           # 超分倍数
        act = nn.ReLU()  # 移除 inplace 参数

        # DIV2K 数据集的 RGB 均值
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)  # 去均值

        # 头部模块（初始特征提取）
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # 主体模块（多个残差组）
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, 
                act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) 
            for _ in range(n_resgroups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))  # 主体输出卷积
        self.body = nn.Sequential(*modules_body)

        # 加均值（用于输出恢复）
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # 输出维度（用于后续解码器，如 LIIF）
        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # 尾部模块（上采样 + 输出卷积）
            modules_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*modules_tail)

    def execute(self, x):
        # x = self.sub_mean(x)  # 可选：减去均值
        x = self.head(x)       # 头部特征提取

        res = self.body(x)     # 主体特征处理
        res += x               # 全局残差连接

        if self.args.no_upsampling:
            x = res  # 不做上采样，直接输出特征
        else:
            x = self.tail(res)  # 上采样并输出
        # x = self.add_mean(x)  # 可选：加上均值
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                try:
                    own_state[name].assign(param)  # Jittor 用 assign 赋值
                except Exception:
                    if name.find('tail') >= 0:
                        print('替换预训练的上采样模块...')
                    else:
                        raise RuntimeError(
                            f"复制参数 {name} 失败，模型维度 {own_state[name].shape}，"
                            f" checkpoint 维度 {param.shape}"
                        )
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(f"state_dict 中存在意外的键: {name}")

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(f"state_dict 中缺失键: {missing}")


@register('rcan')
def make_rcan(n_resgroups=10, n_resblocks=20, n_feats=64, reduction=16,
              scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resgroups = n_resgroups
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.reduction = reduction

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.res_scale = 1
    args.n_colors = 3
    return RCAN(args)