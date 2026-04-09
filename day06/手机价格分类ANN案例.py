
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from torchsummary import summary


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
    # 数据标准化
    # 封装数据集
    train_dataset = TensorDataset(torch.tensor(x_train.values, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(x_test.values, dtype=torch.float32),
                                    torch.tensor(y_test.values, dtype=torch.float32))
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y)) # 特征数，类别数


# todo 2. 搭建ANN
class PhonePriceModel(nn.Module):
    # 初始化模型
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, output_dim)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        # x = torch.softmax(self.linear3(x), dim=1)
        # 使用多分类交叉熵损失函数，不需要softmax
        x = self.output(x)
        return x
# todo 3. 训练模型

def train(train_dataset, input_dim, output_dim):
    # 创建数据加载器
    # 参1是数据集，参2是批次大小，参三是是否打乱数据，训练集打乱，测试集不打乱
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # 创建模型实例
    model = PhonePriceModel(input_dim, output_dim)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 训练模型
    epochs = 50
    for epoch in range(epochs):
        total_loss, batch_num = 0.0, 0
        start = time.time()
        for x, y in train_loader:
            # 切换模型状态
            model.train()
            # 模型预测
            y_pred = model(x)
            loss = criterion(y_pred, y.long())
            # 反向传播和优化
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
        print(f"epoch: {epoch + 1}, loss: {total_loss / batch_num:.4f}, time: {time.time() - start:.2f}s")

    # 保存模型
    # 参1是模型参数包括权重和偏置，参2是保存路径
    # print(f"\n\nmodel.state_dict(): {model.state_dict()}\n\n")
    torch.save(model.state_dict(), './model/phone.pdh')


# todo 4. 评估模型
def evaluate(test_dataset, input_dim, output_dim):
    # 创键网络对象
    model = PhonePriceModel(input_dim, output_dim)
    # 加载模型参数
    model.load_state_dict(torch.load('./model/phone.pdh'))
    # 创建数据加载器 测试集400条
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # 变量
    correct = 0
    for x, y in test_loader:
        model.eval()
        y_pred = model(x)
        # print(f"y_pred: {y_pred}, y: {y}")
        # 获取预测结果的类别, dim = 1表示在行上进行操作，返回每行最大值的索引
        y_pred = torch.argmax(y_pred, dim=1)
        print(f"y_pred.argmax: {y_pred}")
        print(f"y: {y}")

        print(y == y_pred)
        print((y == y_pred).sum())
        correct += (y == y_pred).sum().item()
    print(f"accuracy: {correct / len(test_dataset):.4f}")

if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    # print(f"train_dataset: {train_dataset}, test_dataset: {test_dataset}, input_dim: {input_dim}, output_dim: {output_dim}")
    # model = PhonePriceModel(input_dim, output_dim)

    # 参1是模型，参2是输入数据的维度，批次数和输入特征数
    # summary(model, input_size=(16, input_dim))
    train(train_dataset, input_dim, output_dim)
    evaluate(test_dataset, input_dim, output_dim)