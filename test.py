import argparse
import os
import math
from functools import partial

import yaml
import jittor as jt
from jittor.dataset import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

jt.flags.use_cuda = 1
print("Use CUDA after setting:", jt.flags.use_cuda)  # 应输出 1

def batched_predict(model, inp, coord, cell, bsize):
    with jt.no_grad(): 
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = jt.concat(preds, dim=1)  
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval() 

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    # 数据归一化
    t = data_norm['inp']
    inp_sub = jt.array(t['sub']).view(1, -1, 1, 1)
    inp_div = jt.array(t['div']).view(1, -1, 1, 1)
    t = data_norm['gt']
    gt_sub = jt.array(t['sub']).view(1, 1, -1)
    gt_div = jt.array(t['div']).view(1, 1, -1)

    # 选择评估指标函数
    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        inp = (batch['inp'] - inp_sub) / inp_div
        
        # 推理
        if eval_bsize is None:
            with jt.no_grad():
                pred = model(inp, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(model, inp, batch['coord'], batch['cell'], eval_bsize)
        
        # 还原预测值（反归一化）
        pred = pred * gt_div + gt_sub
        pred = pred.clamp(0, 1)  

        if eval_type is not None:
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape).transpose(0, 3, 1, 2).contiguous()  
            batch['gt'] = batch['gt'].view(*shape).transpose(0, 3, 1, 2).contiguous()

        # 计算指标并累计
        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description(f'val {val_res.item():.4f}')

    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    # 设置可见 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 创建测试数据集和加载器
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        num_workers=8  
    )

    # 加载模型
    model_spec = jt.load(args.model)['model'] 
    model = models.make(model_spec, load_sd=True) 

    # 执行评估
    res = eval_psnr(
        loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True
    )
    print(f'result: {res:.4f}')
    # ========== 新增：先创建目录，再写入文件 ==========
    # 1. 定义结果保存目录和文件路径
    result_dir = 'results'
    task_name = os.path.basename(args.config).replace('test-', '').replace('.yaml', '')  
    save_file = os.path.join(result_dir, 'edsr_liif_tasks_result.txt')

    # 2. 检查并创建目录（如果不存在）
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f'创建目录: {result_dir}')

    # 3. 写入结果到文件
    with open(save_file, 'a') as f:
        f.write(f'{task_name}: {res:.4f}\n')