import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      # 输入
                out_channels=16,    # 输出信道模型
                kernel_size=5,      # 核大小
                stride=1,           # 步大小
                padding=2           # 内边距
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class ResBlock(nn.Module):
    def __init__(self, in_channels=512):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=(1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

    def forward(self, x):
        return self.conv(x)


class BfsCNN(nn.Module):
    def __init__(self):
        super(BfsCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(2, 1),
            ),
            nn.ZeroPad2d(padding=(1, 1, 2, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.conv2 = nn.Sequential(
            ResBlock(64), ResBlock(), ResBlock(), ResBlock(), ResBlock(), ResBlock()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 128, (7, 1)),
            nn.Conv2d(128, 64, (7, 1)),
            nn.Conv2d(64, 32, (7, 1)),
            nn.Conv2d(32, 16, (7, 1)),
            nn.Conv2d(16, 4, (7, 1)),
            nn.Conv2d(4, 2, (7, 1)),
            nn.Conv2d(2, 1, (3, 1))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


if __name__ == '__main__':
    x = torch.randn([10, 1, 201, 99])
    print(x.shape)
    model = BfsCNN()
    output = model(x)
    print(output.shape)
