import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import jittor as jt
from jittor.dataset import Dataset

from datasets import register

# Jittor 不直接支持 transforms，需要自己实现
def to_tensor(pil_img):
    img = np.array(pil_img, dtype=np.float32)
    img = img.transpose(2, 0, 1) / 255.0  # HWC 转 CHW 并归一化
    # Jittor.Var 类型的张量是通道优先的
    return jt.array(img)

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        super().__init__()  
        # 将整个数据集重复多少次,用于增加 epoch 大小
        self.repeat = repeat
        # 缓存模式
        self.cache = cache

        # 决定是读取整个文件夹还是读取划分的训练集测试集
        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        
        # 只加载前 k 个文件, 用于快速调试
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            # 无缓存直接从磁盘加载
            if cache == 'none':
                self.files.append(file)

            # 保存为 pickle 文件
            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                # 转换为 Jittor 的 Var 并缓存
                self.files.append(to_tensor(Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return to_tensor(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = jt.array(x).float() / 255  
            return x

        elif self.cache == 'in_memory':
            return x

# PairedImageFolders 是对ImageFolder的封装, 用于同时处理图像对
@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        super().__init__() 
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
        # Jittor 的 Dataset 本身可以直接设置批处理、线程数等参数，无需像 PyTorch 那样依赖独立的 DataLoader 类
        # 设置批处理大小和工作线程数
        self.set_attrs(
            batch_size=self.dataset_1.batch_size,
            num_workers=self.dataset_1.num_workers
        )

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]    