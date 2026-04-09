"""
    创建线性和随机张量

    线性张量：torch.linspace(start, end, steps) tourch.arange() 从start到end等间隔生成steps个数的张量
"""

import torch
# 线性张量
def dm01():
    t1 = torch.arange(0, 10, 2)     # 0 2 4 6 8 包左不包右
    print(f't1: {t1}, type: {t1.dtype}, shape: {t1.shape}')

    t2 = torch.linspace(0, 10, 5)    # 0 2.5 5 7.5 10 包头包尾
    print(f't2: {t2}, type: {t2.dtype}, shape: {t2.shape}')
    print('-' * 30)



def dm02():

    # torch.initial_seed()
    torch.manual_seed(3)    # 设置随机种子，保证每次生成的随机数相同
    t1 = torch.rand(2, 3)     # 生成一个2行3列的随机张量，元素值在0-1之间
    print(f't1: {t1}, type: {t1.dtype}, shape: {t1.shape}')

    print('-' * 30)


    t2 = torch.randn(2, 3)    # 生成一个2行3列的随机张量，元素值服从标准正态分布
    print(f't2: {t2}, type: {t2.dtype}, shape: {t2.shape}')

    t3 = torch.randint(0, 10, (2, 5))    # 生成一个2行3列的随机张量，元素值在0-10之间的整数
    print(f't3: {t3}, type: {t3.dtype}, shape: {t3.shape}')

if __name__ == '__main__':
    dm02()