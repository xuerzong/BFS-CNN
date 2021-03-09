import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from typing import Any

from src.dataset import BGSDataSet
from src.models import BfsCNN


def train(loader, model, device, loss_fn, op, epoch):
    model.train()
    for i, x in enumerate(loader):
        x, target = x
        x, target = x.to(device), target.to(device)
        output = model(x)
        output = output.view(output.size(0), -1)
        loss = loss_fn(output, target)

        op.zero_grad()
        loss.backward()
        op.step()
        # TODO log
        batch_size = 10
        print("{}/{}".format(i, len(dataset)/batch_size))


if __name__ == '__main__':

    # TODO 使用argparser添加参数指定batch_size等参数

    # model = BfsCNN()
    model = torch.load('model.pkl')
    optimizer = Adam(model.parameters())
    loss_fn = MSELoss()
    dataset = BGSDataSet()
    loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=2)
    epoch = 10
    device = "cpu"
    for i in range(epoch):
        train(loader, model, device, loss_fn, optimizer, i+1)
        # test()
        torch.save(model, 'model.pkl')