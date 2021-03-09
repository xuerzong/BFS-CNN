import torch
from torch.utils.data import Dataset, DataLoader


class BGSDataSet(Dataset):
    """
    TODO 生成模拟数据
    """
    def __init__(self):
        pass

    def __getitem__(self, item):
        """
        根据item返回BGS模拟数据
        :return: Tenor[1, 151, W]
                W
        """
        return torch.randn([1, 151, 100]), torch.zeros([100])

    def __len__(self):
        return 100


if __name__ == '__main__':
    dataset = BGSDataSet()
    loader = DataLoader(dataset=dataset,
                        shuffle=False,
                        batch_size=3,
                        num_workers=2
                        )
    for i, data in enumerate(loader):
        if i % 20 == 0:
            print(data)
            print('{}/{}'.format(i*3, len(dataset)))