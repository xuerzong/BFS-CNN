
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
from test.test import test_test
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

        op.zero_grad()
        loss.backward()
        op.step()

        if i % 10 == 0 or i == len(loader) - 1:
            print("Training {:5d} / {:5d}, loss: {:10.8f}"
                  .format(i + 1, len(loader), train_loss / (i + 1)))

    return float(train_loss) / len(loader)


def test(loader, model, device, loss_fn) -> float:
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
                print("Testing {:5d} / {:5d}, loss: {:10.8f}"
                    .format(i + 1, len(loader), test_loss / (i + 1)))

    return float(test_loss) / len(loader)


if __name__ == '__main__':
    model: nn.Module

    torch.manual_seed(9)
    torch.cuda.manual_seed(9)

    if os.path.exists('model.pkl'):
        model = torch.load('model.pkl')
    else:
        model = BfsCNN()
    # model = BfsCNN()
    N = 128
    batch_size = 16
    learn_rate = 0.001
    epoch = 30

    mean_loss = None

    #torch.manual_seed(9)
    #torch.cuda.manual_seed(9)


    optimizer = Adam(model.parameters(), lr=learn_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    loss_fn = MSELoss()
    dataset = BGSDynamicDataSet(size=480, n=N)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True)
    dataset = BGSDynamicDataSet(size=128, n=N)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)    
    for i in range(epoch):
        
        print("Epoch {} :".format(i + 1))
        #dataset = BGSDynamicDataSet(size=480, n=N)
        #train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True)
        #test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True)

        train(train_loader, model, device, loss_fn, optimizer, mean_loss)
        torch.cuda.empty_cache()
        cur_loss = test(test_loader, model, device, loss_fn)
        print(cur_loss)
        torch.cuda.empty_cache()

        #scheduler.step()

        # if mean_loss is None or mean_loss > cur_loss:
        #    mean_loss = cur_loss
    test(test_loader, model, device, loss_fn)   # torch.save(model, 'model.pkl')
    test_test(model)
    torch.save(model, "model.pkl")
