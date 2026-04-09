import torch
from torch.utils.data import TensorDataset, DataLoader # 数据集对象
from torch import nn                                   # 平方损失函数
from torch import optim                                # 优化器
from sklearn.datasets import make_regression           # 生成线性回归数据集
import matplotlib.pyplot as plt                        # 可视化


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_dataset():
    x, y, coef = make_regression(
        n_samples=100,  # 样本数量
        n_features=1,  # 特征数量
        noise=20, # 噪声水平
        bias=14.5, # 偏置项
        coef=True, # 返回系数
        random_state=3 # 随机数种子
    )

    # 将数据转换为 PyTorch 张量
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return x, y, coef


def train(x, y, coef):
    # 创建数据集对象, tensor->dataset
    dataset = TensorDataset(x, y)
    # 创建数据加载器对象, dataset->dataloader
    # batch_size: 每个批次的样本数量, batch_size: 批次大小, shuffle: 是否打乱数据(训练集打乱, 测试集不打乱)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # 创建线性回归模型对象
    # in_features: 输入特征数量, out_features: 输出特征数量
    model = nn.Linear(in_features=1, out_features=1)

    # 定义损失函数对象
    criterion = nn.MSELoss() # 均方误差损失函数
    # 定义优化器对象
    optimizer = optim.SGD(model.parameters(), lr=0.01) # 随机梯度下降优化器, lr: 学习率
    # 训练模型
    # 定义训练轮数, 每轮的损失, 训练总损失, 样本数
    epochs, loss_list, total_loss, total_samples =  100, [], 0.0, 0
    for epoch in range(epochs):
        for train_x, train_y in dataloader: # 7批, 16 * 6 + 4
            # 模型预测
            y_pred = model(train_x)
            # 计算损失
            loss = criterion(y_pred, train_y.reshape(-1, 1)) # 转换为列向量
            # 计算总损失, 样本批次数
            total_loss += loss.item()
            total_samples += 1
            # 反向传播, 梯度清零, 更新参数
            optimizer.zero_grad()   # 梯度清理
            loss.backward()         # 梯度计算, 反向传播
            optimizer.step()        # 梯度更新
        # 计算平均损失, 添加到损失列表, 打印训练信息
        loss_list.append(total_loss / total_samples)
        print(f"轮数: {epoch + 1}, 平均损失: {loss_list[-1]:.4f}")
    # 打印模型参数
    print(f"{epochs}轮的平均损失分别为: {loss_list}, 模型参数为: {model.weight}, {model.bias}")

    # 可视化损失曲线
    plt.plot(range(epochs), loss_list)
    plt.title('训练损失曲线')
    plt.xlabel('轮数')
    plt.ylabel('平均损失')
    plt.grid()
    plt.show()

    # 可视化数据点和拟合线
    plt.scatter(x, y)
    y_pred = torch.tensor(data=[v * model.weight + model.bias for v in x.detach() ])
    y_true = torch.tensor(data=[v * coef + 14.5 for v in x])

    plt.plot(x, y_pred, color='red', label='拟合线')
    plt.plot(x, y_true, color='green', label='真实线')

    plt.legend()
    plt.grid()

    plt.show()


if __name__ == '__main__':
    x, y, coef = create_dataset()
    print(f"x: {x}, y: {y}, coef: {coef}")
    train(x, y, coef)