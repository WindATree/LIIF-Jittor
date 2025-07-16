# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

from argparse import Namespace

import jittor as jt
import jittor.nn as nn

from models import register


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        )

    def execute(self, x):
        out = self.conv(x)
        return jt.concat((x, out), 1)  

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # 局部特征融合
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def execute(self, x):
        return self.LFF(self.convs(x)) + x  # 残差连接

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        self.args = args
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # RDB 块数量、卷积层数、输出通道数配置
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # 浅层特征提取网络
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # 残差密集块和密集特征融合
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # 全局特征融合
        self.GFF = nn.Sequential(
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        )

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # 上采样网络
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(r),  # 像素重排上采样
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                )
            elif r == 4:
                self.UPNet = nn.Sequential(
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                )
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def execute(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(jt.concat(RDBs_out, 1)) 
        x += f__1  # 全局残差连接

        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)


@register('rdn')
def make_rdn(G0=64, RDNkSize=3, RDNconfig='B',
             scale=2, no_upsampling=False):
    args = Namespace()
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.RDNconfig = RDNconfig

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 3
    return RDN(args)