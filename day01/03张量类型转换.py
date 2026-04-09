"""
    创建指定类型的张量

函数：
    type()
    half()    # 转换为半精度浮点数
    double()  # 转换为双精度浮点数
    float()
    short()
    int()
    long()
"""

import torch

# 创建一个默认类型的张量
t1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)    # 默认是int64
print(f't1: {t1}, type: {t1.dtype}, 张量类型: {type(t1)}')
print('-' * 30)

# 将张量转换为不同类型
t2 = t1.type(torch.int16)    # 转换为int16类型
print(f't2: {t2}, type: {t2.dtype}, 张量类型: {type(t2)}')
print('-' * 30)


print(t2.half())
print(t2.double())
print(t2.float())
print(t2.short())
print(t2.int())
print(t2.long())

