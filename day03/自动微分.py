


import torch

# 定义x, 表示特征
x = torch.ones(2, 5)
print(f"x: {x}")

# 定义y, 表示标签
y = torch.zeros(2, 3)
print(f"y: {y}")


# 初始化权重和偏置
w = torch.randn(5, 3, requires_grad=True)
print(f"w: {w}")
b = torch.randn(3, requires_grad=True)
print(f"b: {b}")

# 前向传播
z = x @ w + b
print(f"z: {z}")

# 定义损失函数
criterion = torch.nn.MSELoss()
loss = criterion(z, y)
print(f"loss: {loss}")
# 自动微分, 反向传播
loss.backward()

print(f"w.grad: {w.grad}")
print(f"b.grad: {b.grad}")

