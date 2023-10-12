import torch
import torch.nn as nn
import torch.nn.functional as F

'''
GAN 模型搭建
生成器模型：
    100噪声(正太分布)
    输入数据集CHW是(1,28,28)，所以输出也是(1,28,28)

判别器模型：
    输入(1,28,28)，输出二分类概率，sigmoid激活
'''


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(in_features=100, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=28*28),
            nn.Tanh()  # 最后一层使用tanh激活，使其分布在(-1,1)之间
        )

    def forward(self, x):
        # print(f'Generator x.shape:{x.shape}')
        x = self.gen(x)
        # print(f'Generator 11 x.shape:{x.shape}')

        x = x.view(-1, 28, 28)
        # print(f'Generator 22 x.shape:{x.shape}')

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(f'Discriminator x.shape:{x.shape}')
        x = x.view(-1, 28 * 28)
        # print(f'Discriminator 11 x.shape:{x.shape}')
        x = self.disc(x)
        # print(f'Discriminator 22 x.shape:{x.shape}')
        return x


if __name__ == '__main__':
    print("GANS")
    generator = Generator()
    discriminator = Discriminator()
