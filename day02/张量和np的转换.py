"""
    张量和np的转换
    .numpy()方法可以将张量转换为numpy数组。
    .numpy().copy()方法可以将张量转换为numpy数组，并且创建一个新的副本，这样修改numpy数组不会影响原始张量。

    np数组转张量
    torch.from_numpy()函数可以将numpy数组转换为张量。共享内存。
    torch.tensor()函数可以将numpy数组转换为张量。创建一个新的副本。不共享内存。

    标量张量转numpy数字
    标量张量可以通过.item()方法转换为Python数字。
"""

import torch
import numpy as np

def dm01():
    t1 = torch.tensor([[1, 2], [3, 4]])
    print(f"原始张量：\n{t1}, 类型: {type(t1)}")

    n1 = t1.numpy()
    print(f"转换为numpy数组：\n{n1}, 类型: {type(n1)}")

    t1[0, 0] = 10

    print(f"修改张量后：\n{t1}, 类型: {type(t1)}")
    print(f"修改张量后对应的numpy数组：\n{n1}, 类型: {type(n1)}")

    n2 = t1.numpy().copy()
    t1[0, 0] = 20
    print(f"修改张量后：\n{t1}, 类型: {type(t1)}")
    print(f"修改张量后对应的numpy数组（副本）：\n{n2}, 类型: {type(n2)}")


def dm02():
    data = np.array([[5, 6], [7, 8]])

    t1 = torch.from_numpy(data) #共享内存
    t2 = torch.tensor(data) #创建副本


    print(f"原始numpy数组：\n{data}, 类型: {type(data)}")
    print(f"使用torch.from_numpy转换为张量：\n{t1}, 类型: {type(t1)}")
    print(f"使用torch.tensor转换为张量：\n{t2}, 类型: {type(t2)}")

    data[0, 0] = 50

    print(f"原始numpy数组：\n{data}, 类型: {type(data)}")
    print(f"使用torch.from_numpy转换为张量：\n{t1}, 类型: {type(t1)}")
    print(f"使用torch.tensor转换为张量：\n{t2}, 类型: {type(t2)}")


def dm03():
    t1 = torch.tensor([42, ])
    print(f"原始标量张量：\n{t1}, 类型: {type(t1)}")
    n1 = t1.item()
    print(f"转换为Python数字：\n{n1}, 类型: {type(n1)}")

if __name__ == '__main__':
    dm03()