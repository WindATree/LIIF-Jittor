import jittor as jt
import jittor.nn as nn

import models
from models import register
from utils import make_coord  # 需确保 utils.make_coord 已转换为 Jittor 版本


@register('metasr')
class MetaSR(nn.Module):

    def __init__(self, encoder_spec):
        super().__init__()

        self.encoder = models.make(encoder_spec)  # 编码器（如 EDSR）已转换为 Jittor 版本
        # 定义 MLP 解码器
        imnet_spec = {
            'name': 'mlp',
            'args': {
                'in_dim': 3,
                'out_dim': self.encoder.out_dim * 9 * 3,
                'hidden_list': [256]
            }
        }
        self.imnet = models.make(imnet_spec)

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)  # 生成编码器特征
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        # 特征展开（3x3 邻域）
        feat = jt.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3]
        )

        # 生成特征图坐标（无需.cuda()）
        feat_coord = make_coord(feat.shape[-2:], flatten=False)
        feat_coord[:, :, 0] -= (2 / feat.shape[-2]) / 2
        feat_coord[:, :, 1] -= (2 / feat.shape[-1]) / 2
        feat_coord = feat_coord.transpose(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        # 坐标调整
        coord_ = coord.copy()  # copy() 替代 clone()
        coord_[:, :, 0] -= cell[:, :, 0] / 2
        coord_[:, :, 1] -= cell[:, :, 1] / 2
        coord_q = (coord_ + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)  # clamp 替代 clamp_

        # 采样特征和坐标
        q_feat = jt.grid_sample(
            feat, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False
        )[:, :, 0, :].transpose(0, 2, 1)  # transpose 替代 permute

        q_coord = jt.grid_sample(
            feat_coord, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False
        )[:, :, 0, :].transpose(0, 2, 1)

        # 相对坐标计算
        rel_coord = coord_ - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2] / 2
        rel_coord[:, :, 1] *= feat.shape[-1] / 2

        # 拼接输入
        r_rev = cell[:, :, 0] * (feat.shape[-2] / 2)
        inp = jt.concat([rel_coord, r_rev.unsqueeze(-1)], dim=-1)  # jt.concat 替代 torch.cat

        # MLP 解码与矩阵乘法
        bs, q = coord.shape[:2]
        pred = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)
        pred = jt.bmm(q_feat.contiguous().view(bs * q, 1, -1), pred)  # jt.bmm 替代 torch.bmm
        pred = pred.view(bs, q, 3)
        return pred

    def execute(self, inp, coord, cell):  # Jittor 用 execute 替代 forward
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)