import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 定义生成器和判别器的网络结构
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # 使用Sigmoid作为生成器的输出激活函数，生成值在[0, 1]之间
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 使用Sigmoid作为判别器的输出激活函数，输出值表示输入数据为真实数据的概率
        )

    def forward(self, x):
        return self.model(x)


# 定义GAN的参数和网络结构
input_dim = 1
output_dim = 1
lr = 0.0002
batch_size = 64
num_epochs = 200

# 创建生成器和判别器
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(input_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 训练GAN
for epoch in range(num_epochs):
    for _ in range(len(data_loader)):
        # 训练判别器
        discriminator.zero_grad()
        real_data = ...  # 加载真实数据
        fake_data = generator(torch.randn(batch_size, input_dim))
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        real_loss = criterion(discriminator(real_data), real_labels)
        fake_loss = criterion(discriminator(fake_data), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        generator.zero_grad()
        fake_data = generator(torch.randn(batch_size, input_dim))
        g_loss = criterion(discriminator(fake_data), real_labels)
        g_loss.backward()
        optimizer_G.step()

# 在训练完成后，您可以使用生成器生成数据点
generated_data = generator(torch.randn(100, input_dim))

# 保存生成器的参数
torch.save(generator.state_dict(), 'generator.pth')
