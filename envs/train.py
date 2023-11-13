import torch
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator

# 定义超参数
input_dim = 128  # 种子特征的维度，初始为128。生成器的随机输入
output_dim = 88  # 用户特征的维度，初始为88
learning_rate = 0.0001  # 学习率为0.0001
batch_size = 64  # 训练批次
epochs = 10000  # 训练周期数
alpha = 1.0  # 超参数，控制熵约束的重要性
beta = 0.1  # 超参数，控制KL散度项的重要性

# 创建生成器和判别器实例
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

# 定义优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# 定义标准损失函数：二元交叉熵损失
criterion = nn.BCELoss()


# 自定义生成器的损失函数
def generator_loss(fake_outputs, real_data, noise):
    # 计算熵项
    entropy_loss = -torch.mean(torch.log(fake_outputs) * fake_outputs)

    # 计算KL散度项
    kl_loss = torch.mean(torch.log(fake_outputs) * fake_outputs - torch.log(real_data) * real_data)

    # 生成器损失包括对抗损失、熵约束和KL散度项
    loss = torch.mean(-torch.log(fake_outputs)) + alpha * entropy_loss - beta * kl_loss
    return loss



# 自定义判别器的损失函数
def discriminator_loss(real_outputs, fake_outputs):
    # 在这里定义自己的判别器损失计算方式
    loss = criterion(real_outputs,fake_outputs)  # 任何你想要的判别器损失计算方法
    return loss


# 训练GAN-SD
for epoch in range(epochs):
    for _ in range(batch_size):
        # 训练判别器
        optimizer_D.zero_grad()
        real_data = ...  # 从真实数据集中获取样本
        noise = torch.randn(batch_size, input_dim)  # 产生噪声数据
        fake_data = generator(noise)  # 产生假样本
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        real_loss = criterion(discriminator(real_data), real_labels)
        fake_loss = criterion(discriminator(fake_data), fake_labels)
        loss_D = real_loss + fake_loss
        loss_D.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        fake_data = generator(torch.randn(batch_size, input_dim))
        fake_outputs = discriminator(fake_data)  # 判别器输出假样本的结果
        loss_G = criterion(fake_outputs, real_labels,noise) # 使用自定义的生成器损失函数
        loss_G.backward()
        optimizer_G.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss D: {loss_D.item()}, Loss G: {loss_G.item()}')

# 保存生成器的训练模型
torch.save(generator.state_dict(), 'generator.pth')
