import argparse
import os
from PIL import Image

import jittor as jt
import jittor.transform as transforms  # Jittor 的数据转换模块

import models
from utils import make_coord  # 确保 make_coord 已转换为 Jittor 版本
from test import batched_predict  # 复用已转换的批量预测函数


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    # 设置可见 GPU（Jittor 自动管理）
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 加载输入图像并转换为 Jittor 张量
    # 替换 torchvision.transforms 为 jittor.transform
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))  # 输出形状：(3, H, W)，值范围 [0,1]

    # 加载模型（Jittor 格式）
    model_spec = jt.load(args.model)['model']  # 用 jt.load 替代 torch.load
    model = models.make(model_spec, load_sd=True)  # 无需 .cuda()，Jittor 自动部署

    # 解析目标分辨率（h, w）
    h, w = list(map(int, args.resolution.split(',')))

    # 生成坐标和单元格大小（适配 Jittor 张量）
    coord = make_coord((h, w))  # 生成 (h*w, 2) 的坐标张量
    cell = jt.ones_like(coord)  # 单元格大小张量
    cell[:, 0] *= 2 / h  # 高度方向单元格大小（归一化到 [-1,1] 范围）
    cell[:, 1] *= 2 / w  # 宽度方向单元格大小

    # 图像预处理（归一化到 [-1,1]）并添加批次维度
    inp = (img - 0.5) / 0.5  # 与原代码一致的归一化
    inp = inp.unsqueeze(0)  # 形状变为 (1, 3, H, W)

    # 批量预测（复用 test.py 中转换的 batched_predict）
    pred = batched_predict(
        model, inp,
        coord.unsqueeze(0),  # 增加批次维度 (1, h*w, 2)
        cell.unsqueeze(0),   # 增加批次维度 (1, h*w, 2)
        bsize=30000
    )[0]  # 取第 0 个样本（移除批次维度）

    # 后处理：反归一化、裁剪范围、调整形状并转换为图像
    pred = (pred * 0.5 + 0.5).clamp(0, 1)  # 从 [-1,1] 转回 [0,1]
    pred = pred.view(h, w, 3)  # 形状 (h, w, 3)
    pred = pred.transpose(2, 0, 1)  # 转换为 (3, h, w)（Jittor 用 transpose 替代 permute）
    pred = pred.cpu().numpy()  # 转换为 numpy 数组

    # 保存输出图像
    transforms.ToPILImage()(pred).save(args.output)