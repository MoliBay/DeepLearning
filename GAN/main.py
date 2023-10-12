import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from IPython import display

from GAN.gans import Generator, Discriminator

'''
利用MNIST手写字母数据集进行基础GAN程序编写
'''

# 噪声生成器
def get_noise(size):
    """
        给生成器准备数据
        - 100维度的向量
    """
    X = torch.randn(size, 100, device=device)
    return X

# 获取真实数据的标签
def get_real_data_labels(size):
    labels = torch.ones(size, 1, device=device)
    return labels

# 获取虚假数据的标签
def get_fake_data_labels(size):
    labels = torch.zeros(size, 1, device=device)
    return labels


root = '../data/MNIST'
Num_Epochs = 100

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
# 加载并预处理图像
data = datasets.MNIST(root="data",
                      train=True,
                      transform=transform,
                      download=True)

# 封装成 DataLoader
data_loader = DataLoader(dataset=data, batch_size=100, shuffle=True)

# 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'

generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 定义一个生成器的优化器
g_optimizer = torch.optim.Adam(params=generator.parameters(), lr=1e-4)
# 定义一个鉴别的优化器
d_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)
# 定义损失函数
loss_fn = nn.BCELoss()
# 定义训练轮次
num_epochs = 1000

# 训练过程
def train(num_epochs):
    num_test_noise = 16
    test_noise = get_noise(num_test_noise)

    for epoch in range(1, num_epochs + 1):

        print(f"当前正在进行 第 {epoch} 轮 ....")

        # 设置训练模式
        generator.train()
        discriminator.train()

        # 遍历真实的图像
        for batch_idx, (batch_real_data, _) in enumerate(data_loader):
            # 1, 先训练鉴别器

            # 1.1 准备数据
            # 图像转向量 [b, 1, 28, 28] ---> [b, 784]
            real_data = batch_real_data.view(batch_real_data.size(0), -1).to(device=device)
            noise = get_noise(real_data.size(0))
            fake_data = generator(noise).detach()

            # 1.2 训练过程

            # 鉴别器的优化器梯度情况
            d_optimizer.zero_grad()

            # 对真实数据鉴别
            real_pred = discriminator(real_data)

            # 计算真实数据的误差
            real_loss = loss_fn(real_pred, get_real_data_labels(real_data.size(0)))

            # 真实数据的梯度回传
            real_loss.backward()

            # 对假数据鉴别
            fake_pred = discriminator(fake_data)

            # 计算假数据的误差
            fake_loss = loss_fn(fake_pred, get_fake_data_labels(fake_data.size(0)))

            # 假数据梯度回传
            fake_loss.backward()

            # 梯度更新
            d_optimizer.step()

            print(f"鉴别器的损失:{real_loss + fake_loss}")

            # 2, 再训练生成器
            # 获取生成器的生成结果
            fake_pred = generator(get_noise(real_data.size(0)))

            # 生产器梯度清空
            g_optimizer.zero_grad()

            # 把假数据让鉴别器鉴别一下
            d_pred = discriminator(fake_pred)

            # 计算损失
            g_loss = loss_fn(d_pred, get_real_data_labels(d_pred.size(0)))

            # 梯度回传
            g_loss.backward()

            # 参数更新
            g_optimizer.step()

            print(f"生成器误差：{g_loss}")

        #  每训练一轮，查看生成器的效果
        generator.eval()

        with torch.no_grad():

            # 正向推理
            img_pred = generator(test_noise)
            img_pred = img_pred.view(img_pred.size(0), 28, 28).cpu().data

            # 画图
            display.clear_output(wait=True)

            # 设置画图的大小
            fig = plt.figure(1, figsize=(12, 8))
            # 划分为 4 x 4 的 网格
            gs = gridspec.GridSpec(4, 4)

            # 遍历每一个
            for i in range(4):
                for j in range(4):
                    # 取每一个图
                    X = img_pred[i * 4 + j, :, :]
                    # 添加一个对应网格内的子图
                    ax = fig.add_subplot(gs[i, j])
                    # 在子图内绘制图像
                    ax.matshow(X, cmap=plt.get_cmap("Greys"))
                    #                 ax.set_xlabel(f"{label}")
                    ax.set_xticks(())
                    ax.set_yticks(())
            plt.show()

train(num_epochs=10)
