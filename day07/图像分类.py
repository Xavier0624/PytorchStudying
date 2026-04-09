import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
import time
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8

# 准备数据集
def create_dataset():
    # 加载训练数据集
    # 参1: 数据存储路径, 参2: 是否为训练集, 参3: 是否下载数据, 参4: 数据预处理方法
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    # 加载测试数据集
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
    return train_dataset, test_dataset


# 搭建CNN网络
class ImageModel(nn.Module):
    # 初始化方法
    def __init__(self):
        super().__init__()

        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        # 池化层1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        # 池化层2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 全连接层1
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        # 全连接层2
        self.fc2 = nn.Linear(120, 84)
        # 输出
        self.output = nn.Linear(84, 10)

    # 前向传播方法
    def forward(self, x):
        # 卷积层1(加权求和) + 激活函数 + 池化层1(降维)
        x = self.pool1(torch.relu(self.conv1(x)))
        # 卷积层2(加权求和) + 激活函数 + 池化层2(降维)
        x = self.pool2(torch.relu(self.conv2(x)))
        # 将多维输入展平为一维
        x = x.reshape(x.size(0), -1)
        # print(f'x.shape: {x.shape}')
        # 全连接层1(加权求和) + 激活函数
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output(x)      # 多分类交叉熵自带了softmax函数
        return x

def train(train_dataset):
    # 创建数据加载器
    dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 创建模型实例
    model = ImageModel()
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 训练模型
    epochs = 10
    for epoch_idx in range(epochs):
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        # 遍历数据加载器
        for x, y in dataloader:
            model.train()
            # 前向传播
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"torch.argmax(y_pred, dim=-1): {torch.argmax(y_pred, dim=-1)}")
            # 统计损失、样本数和正确预测数
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
        print(f"epoch: {epoch_idx + 1}, loss: {total_loss / total_samples:.4f}, accuracy: {total_correct / total_samples:.4f}, time: {time.time() - start:.2f}s")

    torch.save(model.state_dict(), './model/image_model.pth')
def evaluate(test_dataset):
    dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ImageModel()

    model.load_state_dict(torch.load('./model/image_model.pth'))

    total_correct, total_samples = 0, 0
    for x, y in dataloader:
        model.eval()
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=-1)
        total_correct += (y_pred == y).sum().item()
        total_samples += len(y)

    print(f"accuracy: {total_correct / total_samples:.4f}")

if __name__ == '__main__':
    # 获取数据集
    train_dataset, test_dataset = create_dataset()
    # print(f"训练集: {train_dataset.data.shape}")
    # print(f"测试集: {test_dataset.data.shape}")
    # print(f"训练集标签: {train_dataset.class_to_idx}")y
    #
    # # 展示图片
    # plt.figure(figsize=(2, 2))
    # plt.imshow(train_dataset.data[11])
    # plt.title(f"标签: {train_dataset.targets[11]}")
    # plt.show()

    # 搭建模型
    # model = ImageModel()
    # summary(model, input_size=(3, 32, 32), batch_size=BATCH_SIZE)

    # 训练
    # train(train_dataset)
    # 评估
    evaluate(test_dataset)