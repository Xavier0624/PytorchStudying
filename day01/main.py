"""
张量的基本创建方式
"""

# 导入包
import torch
import numpy as np

def dm01():
    # 标量
    t1 = torch.tensor(10)
    print(f't1: {t1}, type: {type(t1)}')
    print('-' * 30)


    data = [[1,2,3], [4, 5, 6]]
    t2 = torch.tensor(data)
    print(f't2: {t2}, type: {type(t2)}')
    print('-' * 30)


    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.tensor(data, dtype=torch.float)
    print(f't3: {t3}, type: {type(t3)}')
    print('-' * 30)

def dm02():
    # 标量
    t1 = torch.Tensor(10)
    print(f't1: {t1}, type: {type(t1)}')
    print('-' * 30)


    data = [[1,2,3], [4, 5, 6]]
    t2 = torch.Tensor(data)
    print(f't2: {t2}, type: {type(t2)}')
    print('-' * 30)


    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.Tensor(data)
    print(f't3: {t3}, type: {type(t3)}')
    print('-' * 30)

    t4 = torch.Tensor(3, 4)
    print(f't4: {t4}, type: {type(t4)}')

def dm03():
    # 标量
    t1 = torch.IntTensor(10)
    print(f't1: {t1}, type: {type(t1)}')
    print('-' * 30)


    data = [[1,2,3], [4, 5, 6]]
    t2 = torch.IntTensor(data)
    print(f't2: {t2}, type: {type(t2)}')
    print('-' * 30)


    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.IntTensor(data)
    print(f't3: {t3}, type: {type(t3)}')
    print('-' * 30)

    data = np.random.randint(0, 10, size=(2, 3))
    t4 = torch.FloatTensor(data)
    print(f't4: {t4}, type: {type(t4)}')
    print('-' * 30)


if __name__ == '__main__':
    dm03()