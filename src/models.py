import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels=512, out_channels=None):
        super(ResBlock, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                padding=(1, 1)
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(1, 1)
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
        )
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512)
        )
        

    def forward(self, x):
        out = self.conv(x)
        x = self.short_cut(x)
        return nn.ReLU()(x + out)


class BFSCNN(nn.Module):
    def __init__(self):
        super(BFSCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(2, 1),
            ),
            nn.ZeroPad2d(padding=(1, 1, 2, 1)),
            nn.BatchNorm2d(64),
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
    x = torch.randn([10, 1, 151, 120])
    print(x.shape)
    model = BFSCNN()
    output = model(x)
    print(output.shape)
