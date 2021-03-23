import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.dataset import BGSDynamicDataSet
import os
import numpy as np
import matplotlib.pyplot as plt
from src.models import BfsCNN

array = np.ndarray


def train(loader, model, device, loss_fn, op):
    model.train()
    train_loss = 0.0
    for i, x in enumerate(loader):
        x, target = x
        x, target = x.to(device), target.to(device)
        output = model(x)
        output = output.view(output.size(0), -1)
        loss = loss_fn(output, target)

        train_loss += loss
        op.zero_grad()
        loss.backward()
        op.step()

        if i % 10 == 0 or i == len(loader) - 1:
            print("Training {:5d} / {:5d}, loss: {:10.5f}"
                  .format(i + 1, len(loader), train_loss / (i + 1)))


def test(loader, model, device, loss_fn):
    model.eval()
    test_loss = 0.0
    for i, x in enumerate(loader):
        x, target = x
        x, target = x.to(device), target.to(device)
        output = model(x)
        output = output.view(output.size(0), -1)
        loss = loss_fn(output, target)
        test_loss += loss

        if i % 10 == 0 or i == len(loader) - 1:
            print("Testing {:5d} / {:5d}, loss: {:10.5f}"
                  .format(i + 1, len(loader), test_loss / (i + 1)))


if __name__ == '__main__':
    model: nn.Module
    if not (os.path.exists('model.pkl')):
        model = BfsCNN()
    else:
        model = torch.load('model.pkl')
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    loss_fn = MSELoss()
    dataset = BGSDynamicDataSet(size=1000)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=4)
    epoch = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for i in range(epoch):
        print("Epoch {} :".format(i + 1))
        train(train_loader, model, device, loss_fn, optimizer)
        test(test_loader, model, device, loss_fn)
        scheduler.step()
    torch.save(model, 'model.pkl')
