import torch
import torch.nn as nn


# 定义生成器类
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()  # 调用父类构造函数，执行模型初始化过程
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),  # 输入维度到128维的全连接层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(128, output_dim),  # 128维到输出维度的全连接层
            nn.Tanh()  # Tanh激活函数
        )

    def forward(self, x):
        return self.model(x)  # 前向传播，输入x通过生成器模型产生生成数据
