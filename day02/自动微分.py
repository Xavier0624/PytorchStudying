import torch

w = torch.tensor(10, requires_grad=True, dtype=torch.float32)

loss = w ** 2 + 20

print(f"初始值：{w}, (0.01 * w.grad) = 无, loss: {loss}")

for i in range(1, 101):
    loss = w ** 2 + 20
    if w.grad is not None:
        w.grad.zero_()


    loss.backward()

    with torch.no_grad():
        w -= 0.01 * w.grad


    print(f"第{i}次迭代：{w}, (0.01 * w.grad) = {0.01 * w.grad}, loss: {loss}")