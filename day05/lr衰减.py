import torch
from torch import optim
import matplotlib.pyplot as plt


def dm01():
    lr = 0.1
    itration = 10
    epochs = 200

    y_true = torch.tensor([0.0])
    x = torch.tensor([1.0], dtype=float)
    w = torch.tensor([1.0], requires_grad=True, dtype=float)

    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 参1优化器对象 参2学习率衰减的周期 参3学习率衰减的倍数
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    lr_list, epoch_list = [], []

    for epoch in range(epochs):
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        for batch in range(itration):
            y_pred = w * x
            criterion = (y_pred - y_true) ** 2 / 2.0
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()
        scheduler.step()
    print(f"lr_list: {lr_list}")
    plt.plot(epoch_list, lr_list)
    plt.show()
def dm02():
    lr = 0.1
    itration = 10
    epochs = 200

    y_true = torch.tensor([0.0])
    x = torch.tensor([1.0], dtype=float)
    w = torch.tensor([1.0], requires_grad=True, dtype=float)

    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 参1优化器对象 参2学习率衰减的周期 参3学习率衰减的倍数
    milestones = [50, 125, 160]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    lr_list, epoch_list = [], []

    for epoch in range(epochs):
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        for batch in range(itration):
            y_pred = w * x
            criterion = (y_pred - y_true) ** 2 / 2.0
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()
        scheduler.step()
    print(f"lr_list: {lr_list}")
    plt.plot(epoch_list, lr_list)
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.legend()
    plt.show()
def dm03():
    lr = 0.1
    itration = 10
    epochs = 200

    y_true = torch.tensor([0.0])
    x = torch.tensor([1.0], dtype=float)
    w = torch.tensor([1.0], requires_grad=True, dtype=float)

    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 参1优化器对象 参2学习率衰减的周期 参3学习率衰减的倍数
    milestones = [50, 125, 160]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    lr_list, epoch_list = [], []

    for epoch in range(epochs):
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        for batch in range(itration):
            y_pred = w * x
            criterion = (y_pred - y_true) ** 2 / 2.0
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()
        scheduler.step()
    print(f"lr_list: {lr_list}")
    plt.plot(epoch_list, lr_list)
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dm03()