import argparse
import os
from PIL import Image
import numpy as np
import jittor as jt
import jittor.transform as transforms

import models
from utils import make_coord  
from test import batched_predict  

jt.flags.use_cuda = 1
print("Use CUDA after setting:", jt.flags.use_cuda)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 1. 加载输入图像（确保 dtype=float32）
    pil_img = Image.open(args.input).convert('RGB')
    img_np = np.array(pil_img).transpose(2, 0, 1).astype(np.float32)  # float32
    img_np = img_np / 255.0  # [0,1]
    img = jt.array(img_np)  # 自动在 GPU 上（因 use_cuda=1）

    # 2. 加载模型（无需 to_device，由 jt.flags.use_cuda 控制）
    model_spec = jt.load(args.model)['model'] 
    model = models.make(model_spec, load_sd=True)  # 模型自动在 GPU 上

    # 3. 生成坐标和单元格（确保在 GPU 上）
    h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w))  
    cell = jt.ones_like(coord)
    cell[:, 0] *= 2 / h  
    cell[:, 1] *= 2 / w  

    # 4. 输入处理
    inp = (img - 0.5) / 0.5  
    inp = inp.unsqueeze(0)  # 已在 GPU 上

    # 5. 批量预测（优化内存）
    pred = batched_predict(
        model, inp,
        coord.unsqueeze(0),
        cell.unsqueeze(0),
        bsize=1000  
    )[0]
    print("Pred shape before post-processing:", pred.shape)  # 应输出 (3, h, w)
    
    # 6. 后处理（尽早移到 CPU 释放 GPU 内存）
    pred = (pred * 0.5 + 0.5).clamp(0, 1)  # [0,1]
    pred_np = pred.cpu().numpy()  # 转为 numpy (3, h, w)

    print("Pred shape after numpy:", pred_np.shape)  # 确认形状为 (3, h, w)

    # 转换为 (h, w, 3) 格式
    pred_np = pred_np.reshape(h, w, 3)  # 关键修正：用 reshape 而非 transpose
    print("Pred shape after transpose:", pred_np.shape)  # 应输出 (h, w, 3)

    # 确保数值范围和类型正确
    print("Value range before scaling:", pred_np.min(), pred_np.max())  # 应在 [0,1]
    pred_np = (pred_np * 255).astype(np.uint8)  # 转为 0-255 的 uint8

    # 保存图像
    Image.fromarray(pred_np).save(args.output)
    print(f"Saved to {args.output}")

   