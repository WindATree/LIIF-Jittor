import os
import time
import shutil
import math

import jittor as jt
import numpy as np
from jittor.optim import SGD, Adam  # Jittor 优化器
from tensorboardX import SummaryWriter  


class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return f'{t / 3600:.1f}h'
    elif t >= 60:
        return f'{t / 60:.1f}m'
    else:
        return f'{t:.1f}s'


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input(f'{path} exists, remove? (y/[n]): ') == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    """计算模型参数数量"""
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return f'{tot / 1e6:.1f}M'
        else:
            return f'{tot / 1e3:.1f}K'
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """生成网格中心坐标"""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * jt.arange(n)
        coord_seqs.append(seq)
    ret = jt.stack(jt.meshgrid(*coord_seqs), dim=-1) 
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """将图像转换为坐标 - RGB对"""
    # img 形状 (3, H, W)
    coord = make_coord(img.shape[-2:])  # 生成坐标
    rgb = img.view(3, -1).transpose(1, 0)  # 形状 (H*W, 3)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    # sr: 超分结果，hr: 高分辨率图像
    diff = (sr - hr) / rgb_range  # 归一化

    # 根据数据集裁剪边界
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:  # 彩色图像转灰度
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = jt.array(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)  
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff

    # 计算 MSE 和 PSNR
    mse = valid.pow(2).mean()
    log10_mse = jt.log(mse) / jt.log(jt.float32(10))  
    return -10 * log10_mse