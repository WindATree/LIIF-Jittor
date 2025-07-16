import jittor as jt
import jittor.nn as nn

import models
from models import register
from utils import make_coord  # 需确保 utils.make_coord 已转换为 Jittor 版本


@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)  # 编码器（如 EDSR）已转换为 Jittor 版本
        
        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9  # 3x3 邻域特征展开
            imnet_in_dim += 2  # 附加坐标信息
            if self.cell_decode:
                imnet_in_dim += 2  # 附加细胞尺寸信息
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})  # MLP 解码器
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)  # 生成编码器特征
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            # 无解码器时直接用最近邻采样
            ret = jt.nn.grid_sample(
                feat, coord.flip(-1).unsqueeze(1),  # coord 翻转维度以匹配 grid_sample 要求
                mode='nearest', align_corners=False
            )[:, :, 0, :].transpose(0, 2, 1)  # Jittor 用 transpose 替代 permute
            return ret

        # 特征展开（3x3 邻域）
        if self.feat_unfold:
            # 保存 unfold 前的原始形状（4 维：B, C, H, W）
            B, C, H, W = feat.shape  # 关键：这里的 H 和 W 是原始特征的空间维度
            # 执行 unfold（输出 3 维：B, C*9, H*W）
            feat = jt.nn.unfold(feat, kernel_size=3, padding=1)
            # 用原始 H 和 W 重塑形状（恢复为 4 维：B, C*9, H, W）
            feat = feat.view(B, C*9, H, W).contiguous()
            
        # 局部集成参数
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # 计算特征图坐标范围
        rx = 2 / feat.shape[-2] / 2  # 高度方向单位长度
        ry = 2 / feat.shape[-1] / 2  # 宽度方向单位长度

        # 生成特征图自身坐标（无需.cuda()，Jittor自动管理设备）
        feat_coord = make_coord(feat.shape[-2:], flatten=False) \
            .transpose(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # 坐标偏移（局部集成）
                coord_ = coord.copy()  # Jittor 用 copy() 替代 clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_ = coord_.clamp(-1 + 1e-6, 1 - 1e-6)  # Jittor 无 in-place 操作，用 clamp 替代 clamp_

                # 采样特征和坐标
                q_feat = jt.nn.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False
                )[:, :, 0, :].transpose(0, 2, 1)  # transpose 替代 permute

                q_coord = jt.nn.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False
                )[:, :, 0, :].transpose(0, 2, 1)

                # 相对坐标计算
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # 拼接特征和坐标
                inp = jt.concat([q_feat, rel_coord], dim=-1)  # Jittor 用 jt.concat 替代 torch.cat

                # 若启用细胞解码，拼接细胞信息
                if self.cell_decode:
                    rel_cell = cell.copy()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = jt.concat([inp, rel_cell], dim=-1)

                # MLP 解码
                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                # 计算权重面积
                area = jt.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])  # jt.abs 替代 torch.abs
                areas.append(area + 1e-9)

        # 加权融合预测结果
        tot_area = jt.stack(areas).sum(dim=0)  # jt.stack 替代 torch.stack
        if self.local_ensemble:
            # 交换面积顺序（保持与原逻辑一致）
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret += pred * (area / tot_area).unsqueeze(-1)
        return ret

    def execute(self, inp, coord, cell):  # Jittor 用 execute 替代 forward
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)