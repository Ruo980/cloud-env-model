import torch
import torch.nn as nn

# 定义判别器类
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),  # 输入维度到128维的全连接层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(128, 1),  # 128维到输出维度1的全连接层
            nn.Sigmoid()  # Sigmoid激活函数，输出范围在0到1之间
        )

    def forward(self, x):
        return self.model(x)  # 前向传播，输入x通过判别器模型产生判别结果
