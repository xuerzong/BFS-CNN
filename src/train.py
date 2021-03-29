
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


def train(loader, model, device, loss_fn, op) -> array:
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
        op.zero_grad()
        loss.backward()
        op.step()
        if i % 10 == 0 or i == len(loader) - 1:
            print("Training {:5d} / {:5d}, loss: {:10.5f}"
                  .format(i + 1, len(loader), train_loss / (i + 1)))

    return loss_arr


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
    N = 128
    batch_size = 4
    learn_rate = 0.001
    epoch = 22
    train_csv_name = f'x_train_{batch_size}_{learn_rate}'
    test_csv_name = f'x_test_{batch_size}_{learn_rate}'

    optimizer = Adam(model.parameters(), lr=learn_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    loss_fn = MSELoss()
    dataset = BGSDynamicDataSet(size=375, n=N)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataset = BGSDynamicDataSet(size=128, n=N)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for i in range(epoch):
        print("Epoch {} :".format(i + 1))

        train_loss_list_tmp = train(train_loader, model, device, loss_fn, optimizer)
        torch.cuda.empty_cache()
        test_loss_list_tmp = test(test_loader, model, device, loss_fn)
        torch.cuda.empty_cache()

        scheduler.step()

        write_csv(filename=train_csv_name, flag=False, data=train_loss_list_tmp)
        write_csv(filename=test_csv_name, flag=False, data=test_loss_list_tmp)

        torch.save(model, 'model_x.pkl')

