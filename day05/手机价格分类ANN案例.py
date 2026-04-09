
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


# todo 1. 构建数据集
def create_dataset():
    # 加载数据文件
    data = pd.read_csv('./data/train.csv')
    # print(f"data:{data.head()}")
    # print(f"data.shape:{data.shape}")
    # 获取特征列和标签列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # print(f"x.head: {x.head()}, x.shape: {x.shape}")
    # print(f"y.head: {y.head()}, y.shape: {y.shape}")
    # 转换浮点型
    x = x.astype(float)
    # print(f"x.head: {x.head()}, x.shape: {x.shape}")
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)
    # 封装数据集
    train_dataset = TensorDataset(torch.tensor(x_train.values, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(x_test.values, dtype=torch.float32),
                                    torch.tensor(y_test.values, dtype=torch.float32))
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y)) # 特征数，类别数


# todo 2. 搭建ANN


# todo 3. 训练模型


# todo 4. 评估模型


if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    print(f"train_dataset: {train_dataset}, test_dataset: {test_dataset}, input_dim: {input_dim}, output_dim: {output_dim}")