import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.dataset import BGSDynamicDataSet
from src.models import BfsCNN


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

        if i % 100 == 0 or i == len(loader) - 1:
            print("Training {:5d} / {:5d}, loss: {:10.5f}".format(i + 1, len(loader), train_loss / len(loader)))


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

        if i % 100 == 0 or i == len(loader) - 1:
            print("Testing {:5d} / {:5d}, loss: {:10.5f}".format(i + 1, len(loader), test_loss / len(loader)))


if __name__ == '__main__':

    # TODO 使用argparser添加参数指定batch_size等参数

    model: nn.Module = BfsCNN()
    # model = torch.load('model.pkl')
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    loss_fn = MSELoss()
    dataset = BGSDynamicDataSet()
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
