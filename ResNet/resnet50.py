import torch
from torch import nn

'''
ResNet50
'''


# 定义卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()

        expansion = 4
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=expansion * out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(expansion * out_channels)
        )
        self.shutCut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=expansion*out_channels,
                      kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(expansion*out_channels)
        )

    def forward(self, x):
        print(f'ConvBlock x.shape:{x.shape}')

        x1 = self.body(x)
        x2 = self.shutCut(x)
        out = nn.functional.relu(x1+x2)
        print(f'ConvBlock x1.shape:{x1.shape}')
        print(f'ConvBlock x2.shape:{x2.shape}')
        print(f'ConvBlock out.shape:{out.shape}')

        return out

# 定义直连块
class DirectBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DirectBlock, self).__init__()

        expansion = 4
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=expansion * out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(expansion * out_channels)
        )

    def forward(self, x):
        print(f'DirectBlock x.shape:{x.shape}')
        x = self.body(x) + x
        x = nn.functional.relu(x)
        print(f'DirectBlock x.shape:{x.shape}')
        return x


class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.body = nn.Sequential(
            # stage1
            nn.Sequential(
                ConvBlock(in_channels=64, out_channels=64, stride=1),
                DirectBlock(in_channels=256, out_channels=64),
                DirectBlock(in_channels=256, out_channels=64)
            ),
            # stage2
            nn.Sequential(
                ConvBlock(in_channels=256, out_channels=128, stride=2),
                DirectBlock(in_channels=512, out_channels=128),
                DirectBlock(in_channels=512, out_channels=128),
                DirectBlock(in_channels=512, out_channels=128),
            ),
            # stage3
            nn.Sequential(
                ConvBlock(in_channels=512, out_channels=256, stride=2),
                DirectBlock(in_channels=1024, out_channels=256),
                DirectBlock(in_channels=1024, out_channels=256),
                DirectBlock(in_channels=1024, out_channels=256),
                DirectBlock(in_channels=1024, out_channels=256),
                DirectBlock(in_channels=1024, out_channels=256),
            ),
            # stage4
            nn.Sequential(
                ConvBlock(in_channels=1024, out_channels=512, stride=2),
                DirectBlock(in_channels=2048, out_channels=512),
                DirectBlock(in_channels=2048, out_channels=512),
            )
        )

        self.foot = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=2048, out_features=1000)
        )

    def forward(self, x):
        x = self.head(x)
        print(f'head out x.shape:{x.shape}')
        x = self.body(x)
        x = self.foot(x)
        return x



if __name__ == '__main__':

    print(f'ResNet50')
    resnet = ResNet50()
    images = torch.randn(2, 3, 224, 224)
    out = resnet(images)
    print(f'resnet out.shape:{out.shape}')
