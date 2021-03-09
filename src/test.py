import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./data/',
    train=True,
    transform=torchvision.transforms.ToTensor(),     # (0, 255) => (0, 1)
    download=DOWNLOAD_MNIST
)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
test_data = torchvision.datasets.MNIST(root='./data/', train=False)

test_x = Variable(
    torch.unsqueeze(test_data.test_data, dim=1),
    volatile=True
).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]


class BFSCNN(nn.Module):
    def __init__(self):
        super(BFSCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
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
        return output, x    # return x for visualization


# -- --
BC = BFSCNN()

optimizer = torch.optim.Adam(BC.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

try:
    from sklearn.manifold import TSNE; HAS_SK = True
except:
    HAS_SK = False; print('Please install sklearn for layer visualization')
