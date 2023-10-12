import torch
from torch import nn
'''
定义模型：VggNet 16
'''


class VggNet(nn.Module):
    def __init__(self):
        super(VggNet, self).__init__()
        self.stage1 = nn.Sequential(
            # 第1层卷积
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            # 第2层卷积
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        # 第1个池化层
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage2 = nn.Sequential(
            # 第1层卷积
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            # 第2层卷积
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        # 第2个池化层
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage3 = nn.Sequential(
            # 第1层卷积
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),

            # 第2层卷积
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),

            # 第3层卷积
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        # 第3个池化层
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage4 = nn.Sequential(
            # 第1层卷积
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),

            # 第2层卷积
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),

            # 第3层卷积
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        # 第4个池化层
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage5 = nn.Sequential(
            # 第1层卷积
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),

            # 第2层卷积
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),

            # 第3层卷积
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        # 第5个池化层
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage6 = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7,
                      out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096,
                      out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096,
                      out_features=10)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.mp1(x)
        x = self.stage2(x)
        x = self.mp2(x)
        x = self.stage3(x)
        x = self.mp3(x)
        x = self.stage4(x)
        x = self.mp4(x)
        x = self.stage5(x)
        x = self.mp5(x)
        x = x.view(x.size(0), -1)
        x = self.stage6(x)
        return x


if __name__ == '__main__':
    print('VggNet')
    model = VggNet()
    imgs = torch.randn(2, 3, 224, 224)
    out = model(imgs)
    print(out.shape)
    print(out)
