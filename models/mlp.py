import jittor as jt
import jittor.nn as nn

from models import register


@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))  # Jittor 线性层，接口与 PyTorch 一致
            layers.append(nn.ReLU())  # Jittor ReLU 激活函数
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)  # 保持 Sequential 容器用法

    def execute(self, x):  # Jittor 用 execute 替代 forward
        shape = x.shape[:-1]  # 保留除最后一维外的形状（用于后续恢复）
        x = self.layers(x.view(-1, x.shape[-1]))  # 展平为二维张量输入网络
        return x.view(*shape, -1)  # 恢复原始形状（除最后一维替换为输出维度）