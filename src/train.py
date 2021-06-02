import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.dataset import BGSDynamicDataSet
import os
import numpy as np
from src.models import BFSCNN

array = np.ndarray


def train_model(
    loader: DataLoader,
    model: BFSCNN,
    device: str,
    loss_function: MSELoss,
    optimizer: Adam
) -> float:

    """训练模型
    :param loader: 数据加载器
    :param model: BFSCNN模型
    :param device: 'cuda' 或者 'cpu'
    :param loss_function: 损失函数 MSELoss损失函数
    :param optimizer: 优化函数 Adam优化函数
    :returns: 训练函数的平均损失值
    """

    model.train()
    train_loss = None
    for i, x in enumerate(loader):
        x, target = x
        x, target = x.to(device), target.to(device)
        output = model(x)
        output = output.view(output.size(0), -1)
        loss = loss_function(output, target)

        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 or i == len(loader) - 1:
            print("Training {:5d} / {:5d}, loss: {:10.8f}"
                  .format(i + 1, len(loader), train_loss / (i + 1)))

    return float(train_loss) / len(loader)


def inspect_model(
    loader: DataLoader,
    model: BFSCNN,
    device: str,
    loss_fn: MSELoss
) -> float:

    """验证模型
    :param loader: 数据加载器
    :param model: BFSCNN模型
    :param device: 'cuda' 或者 'cpu'
    :param loss_function: 损失函数 MSELoss损失函数
    :returns: 平均损失值
    """

    model.eval()
    inspect_loss = 0

    with torch.no_grad():
        for i, x in enumerate(loader):
            x, target = x
            x, target = x.to(device), target.to(device)
            output = model(x)
            output = output.view(output.size(0), -1)
            loss = loss_fn(output, target)

            inspect_loss += loss

            if i % 10 == 0 or i == len(loader) - 1:
                print("Testing {:5d} / {:5d}, loss: {:10.8f}"
                    .format(i + 1, len(loader), inspect_loss / (i + 1)))

    return float(inspect_loss) / len(loader)


if __name__ == '__main__':

    model: BFSCNN

    if os.path.exists('model.pkl'):
        model = torch.load('model.pkl')
    else:
        model = BFSCNN()
    N = 224
    batch_size = 8
    learn_rate = 0.001
    epoch = 22

    mean_loss = None


    optimizer = Adam(model.parameters(), lr=learn_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    loss_fn = MSELoss()


    dataset = BGSDynamicDataSet(size=480, n=N)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for i in range(epoch):
        
        print(f'{i + 1} / {epoch} epoch:')

        
        # 训练模型
        train_model(data_loader, model, device, loss_fn, optimizer)
        torch.cuda.empty_cache()

        # 验证模型
        cur_loss = inspect_model(data_loader, model, device, loss_fn)
        torch.cuda.empty_cache()


        scheduler.step()

        if mean_loss is None or mean_loss > cur_loss:
           mean_loss = cur_loss
           torch.save(model, 'model.pkl')
