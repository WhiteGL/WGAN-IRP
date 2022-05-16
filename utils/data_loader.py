import torch
from torch.utils.data import Dataset
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
from utils.recurrence import intertemporal_recurrence_matrix


class TSDataset(Dataset):
    """Time series dataset."""

    def __init__(self, csv_file, value_col, time_window, is_seq=True, normalize=True):
        """
        Args:
            csv_file (string): path to csv file
            value_col: name of the column containing values
            time_window: time window to consider for conditioning/generation
            is_seq: True连续取序列或False间隔取序列
            normalize (bool): whether to normalize the data in [-1,1]
        """

        # 获取指定列的数据
        if value_col == 'None':
            df = pd.read_csv(csv_file, header=None)
            arr = np.asarray(df, dtype=np.float32)
        else:
            df = pd.read_csv(csv_file)
            df = df.filter([value_col], axis=1)
            df.rename(columns={value_col: "Value"}, inplace=True)
            # 连续取序列或间隔取序列
            if is_seq:
                value = df.Value
                arr = np.asarray([value[i:i + time_window] for i in range(len(df) - time_window)], dtype=np.float32)
            else:
                n = (len(df) // time_window) * time_window
                value = df.Value
                arr = np.asarray([value[time_window * i:time_window * i + time_window] for i in range(n // time_window)]
                                 , dtype=np.float32)
        # # minmax归一化，将所有数据映射至(0,1)区间，加上一个极小值避免0值无法运算问题
        # # data = self.normalize(arr) if normalize else arr
        # length = arr.shape[0]
        # self.data = torch.empty(length, 1, time_window, time_window)
        # # 计算IRP值，该值域为(-∞，+∞)，记录准确值域
        # for i in range(length):
        #     matrix = torch.from_numpy(intertemporal_recurrence_matrix(arr[i]))
        #     self.data[i] = matrix.view((1, time_window, time_window))
        # self.min = self.data.min()
        # self.max = self.data.max()
        # # 再一次线性映射，将值域限制在(-1,1)内
        #————————————————————————————————
        # data = self.normalize(arr) if normalize else arr
        length = arr.shape[0]
        self.data = torch.empty(length, 1, time_window, time_window)
        for i in range(length):
            matrix = torch.from_numpy(intertemporal_recurrence_matrix(arr[i]))
            # matrix = self.normalize(matrix)
            self.data[i] = matrix.view(1, time_window, time_window)
        self.data = self.data[:50]

        # do a tanh as normalization
        # self.data = torch.tanh(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def normalize(self, x):
        """Normalize input in [0,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()
        return (x - self.min) / (self.max - self.min) + 1e-7

    def denormalize(self, x):
        """Revert [0,1] normalization"""
        if not hasattr(self, 'max') or not hasattr(self, 'min'):
            raise Exception("You are calling denormalize, but the input was not normalized")
        return 0.5 * (x * self.max - x * self.min + self.max + self.min)


def get_data_loader(path, value_col, time_window, batch_size, normalize=True, shuffle=True):

    dataset = TSDataset(path, value_col, time_window, normalize)
    train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print('got dataloader')
    return train_loader
