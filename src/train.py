
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.dataset import BGSDynamicDataSet
import os
import numpy as np
from src.models import BfsCNN
from src.utils import write_csv

array = np.ndarray


def train(loader, model, device, loss_fn, op, mean_loss) -> float:
    model.train()
    train_loss = 0.0

    loss_arr = np.array([], dtype=float)

    for i, x in enumerate(loader):
        x, target = x
        x, target = x.to(device), target.to(device)
        output = model(x)
        output = output.view(output.size(0), -1)
        loss = loss_fn(output, target)

        loss_arr = np.append(loss_arr, float(loss))

        train_loss += loss

        if mean_loss == 0:
            print(1)
            op.zero_grad()
            loss.backward()
            op.step()
            train_loss += loss
        elif mean_loss >= float(loss):
            print(2)
            op.zero_grad()
            loss.backward()
            op.step()
            train_loss += loss
        else:
            print(3)
            train_loss += mean_loss
        
        print(float(loss))

        if i % 10 == 0 or i == len(loader) - 1:
            print("Training {:5d} / {:5d}, loss: {:10.5f}"
                  .format(i + 1, len(loader), train_loss / (i + 1)))

    return train_loss / (i + 1)


def test(loader, model, device, loss_fn) -> array:
    model.eval()
    test_loss = 0.0

    loss_arr = np.array([], dtype=float)
    with torch.no_grad():
        for i, x in enumerate(loader):
            x, target = x
            x, target = x.to(device), target.to(device)
            output = model(x)
            output = output.view(output.size(0), -1)
            loss = loss_fn(output, target)
            test_loss += loss

            loss_arr = np.append(loss_arr, float(loss))

            if i % 10 == 0 or i == len(loader) - 1:
                print("Testing {:5d} / {:5d}, loss: {:10.5f}"
                    .format(i + 1, len(loader), test_loss / (i + 1)))

    return loss_arr


if __name__ == '__main__':
    model: nn.Module
    if os.path.exists('model.pkl'):
        model = torch.load('model.pkl')
    else:
        model = BfsCNN()
    model = BfsCNN()
    N = 224
    batch_size = 8
    learn_rate = 0.001
    epoch = 30

    mean_loss = 0


    optimizer = Adam(model.parameters(), lr=learn_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    loss_fn = MSELoss()
    dataset = BGSDynamicDataSet(size=480, n=N)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataset = BGSDynamicDataSet(size=128, n=N)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for i in range(epoch):
        print("Epoch {} :".format(i + 1))

        mean_loss = train(train_loader, model, device, loss_fn, optimizer, mean_loss)
        torch.cuda.empty_cache()
        test_loss_list_tmp = test(test_loader, model, device, loss_fn)
        torch.cuda.empty_cache()

        scheduler.step()

        torch.save(model, 'model.pkl')
    

    torch.save(model, 'xxx.pkl')

