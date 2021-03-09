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
