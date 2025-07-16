""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

import argparse
import os

import yaml
import jittor as jt
import jittor.nn as nn
from tqdm import tqdm
from jittor.dataset import DataLoader  # Jittor 的 DataLoader
from jittor.lr_scheduler import MultiStepLR  # Jittor 的学习率调度器

import datasets
import models
import utils
from test import eval_psnr  # 需确保 eval_psnr 已转换为 Jittor 版本


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    # 创建数据集（已转换为 Jittor 版本的 datasets）
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    # Jittor DataLoader，参数与 PyTorch 类似但无需 pin_memory
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=(tag == 'train'),
        num_workers=8  # Jittor 支持多线程加载
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        # 加载 Jittor 模型权重（替换 torch.load）
        sv_file = jt.load(config['resume'])
        # 构建模型（无需 .cuda()，Jittor 自动管理设备）
        model = models.make(sv_file['model'], load_sd=True)
        # 构建优化器（已转换为 Jittor 版本的 utils.make_optimizer）
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            # Jittor 的 MultiStepLR 用法与 PyTorch 一致
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        # 恢复调度器状态
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        # 初始化模型（无需 .cuda()）
        model = models.make(config['model'])
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer,** config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()  # Jittor 模型训练模式
    loss_fn = nn.L1Loss()  # Jittor 的 L1 损失
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    # 数据归一化参数（替换 torch.FloatTensor 为 jt.array，无需 .cuda()）
    t = data_norm['inp']
    inp_sub = jt.array(t['sub']).view(1, -1, 1, 1)
    inp_div = jt.array(t['div']).view(1, -1, 1, 1)
    t = data_norm['gt']
    gt_sub = jt.array(t['sub']).view(1, 1, -1)
    gt_div = jt.array(t['div']).view(1, 1, -1)

    for batch in tqdm(train_loader, leave=False, desc='train'):
        # Jittor 张量无需 .cuda()，自动在设备上运行
        inp = (batch['inp'] - inp_sub) / inp_div
        pred = model(inp, batch['coord'], batch['cell'])

        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)

        train_loss.add(loss.item())

        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播（Jittor 自动微分）
        optimizer.step()  # 更新参数

        # 释放中间变量（Jittor 自动管理内存，可选）
        pred = None; loss = None

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    # 多 GPU 支持（Jittor 的 DataParallel）
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        # 记录学习率
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # 训练一轮
        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        # 模型状态保存（区分多 GPU 情况）
        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()  # Jittor 模型状态字典
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        # 保存模型（Jittor 保存用 jt.save）
        jt.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            jt.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        # 验证
        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            # 评估 PSNR（需确保 eval_psnr 已转换为 Jittor 版本）
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                jt.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        # 计时信息
        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    # 设置可见 GPU（Jittor 会自动使用）
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    # 构建保存路径
    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)