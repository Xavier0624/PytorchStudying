
import torch
import torch.nn as nn
from torchsummary import summary
# todo: 1.搭建神经网络
class ModelDemo(nn.Module):
    # 魔法方法init
    def __init__(self):
        super().__init__()

        # 搭建隐藏层和输出层
        # 隐藏层1: 输入特征3，输出特征3
        self.linear1 = nn.Linear(3, 3)
        # 隐藏层2: 输入特征3，输出特征2
        self.linear2 = nn.Linear(3, 2)
        # 输出层: 输入特征2，输出特征2
        self.output = nn.Linear(2, 2)

        # 初始化
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.kaiming_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)


    # 前向传播
    def forward(self, x):
        # 隐藏层1: 加权求和，激活函数
        # x = self.linear1(x) # 加权求和
        # x = torch.sigmoid(x) # 激活函数

        x = torch.sigmoid(self.linear1(x))


        # 隐藏层2: 加权求和，激活函数
        x = torch.relu(self.linear2(x))

        # 输出层
        x = torch.softmax(self.output(x), dim=-1) # dim = -1 表示按行计算

        # 返回输出
        return x

# todo: 2.模型预测
def train():
    # 创建实例
    model = ModelDemo()
    data = torch.randn(size=(5, 3))
    print(f"data: {data}, shape: {data.shape}, requires_grad: {data.requires_grad}")
    # 模型预测
    output = model(data)
    print(f"output: {output}, shape: {output.shape}, requires_grad: {output.requires_grad}")


    summary(model, input_size=(5, 3))

    for name, param in model.named_parameters():
        print(f"name: {name}, param: {param.data}, requires_grad: {param.requires_grad}")

if __name__ == '__main__':
    train()
