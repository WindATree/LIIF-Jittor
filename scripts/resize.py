import os
from PIL import Image
from tqdm import tqdm

# 将 1024*1024 缩放到 [256, 128, 64, 32]
for size in [256, 128, 64, 32]:
    # size = 256 从原始 1024 开始缩放
    if size == 256:
        inp = './data1024x1024'
    # 否则从 256 开始缩放
    else:
        inp = './256'
    print(size)
    os.mkdir(str(size))
    filenames = os.listdir(inp)
    # 缩放每张图片并保存
    for filename in tqdm(filenames):
        Image.open(os.path.join(inp, filename)) \
            .resize((size, size), Image.BICUBIC) \
            .save(os.path.join('.', str(size), filename.split('.')[0] + '.png'))
