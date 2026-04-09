import torch
import torch.nn as nn

def dm01():
    w = torch.tensor([1.0], requires_grad=True, dtype=float)

    criterion = (w ** 2 / 2.0)
    # SGD: Stochastic Gradient Descent随机梯度下降，lr是学习率，momentum是动量，动量可以加速SGD在相关方向上的收敛，并抑制振荡。
    optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)
    # 梯度清理 反向传播 更新权重
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w.data}, grad: {w.grad} ")

    criterion = (w ** 2 / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w.data}, grad: {w.grad} ")

def dm02():
    w = torch.tensor([1.0], requires_grad=True, dtype=float)

    criterion = (w ** 2 / 2.0)
    optimizer = torch.optim.Adagrad([w], lr=0.01)
    # 梯度清理 反向传播 更新权重
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w.data}, grad: {w.grad} ")

    criterion = (w ** 2 / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w.data}, grad: {w.grad} ")
def dm03():
    w = torch.tensor([1.0], requires_grad=True, dtype=float)

    criterion = (w ** 2 / 2.0)
    optimizer = torch.optim.RMSprop([w], lr=0.01)
    # 梯度清理 反向传播 更新权重
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w.data}, grad: {w.grad} ")

    criterion = (w ** 2 / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w.data}, grad: {w.grad} ")
def dm04():
    w = torch.tensor([1.0], requires_grad=True, dtype=float)

    criterion = (w ** 2 / 2.0)
    optimizer = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.999))
    # 梯度清理 反向传播 更新权重
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w.data}, grad: {w.grad} ")

    criterion = (w ** 2 / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w.data}, grad: {w.grad} ")
if __name__ == '__main__':
    dm04()