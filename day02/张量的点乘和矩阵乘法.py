"""
点乘：张量的维度一致，对应元素进行乘法
矩阵：列，行相同

t1 * t2, t1.mul(t2)

t1 @ t2, t1.matmul(t2), t1.dot(t2) 第三个只对一维张量有效
"""


import torch

def dm01():
    t1 = torch.tensor([[1, 2], [3, 4]])
    t2 = torch.tensor([[5, 6], [7, 8]])
    t3 = t1 * t2
    t4 = t1.mul(t2)

    print(f"t3: {t3}")
    print(f"t4: {t4}")

    t5 = t1 @ t2
    print(f"t5: {t5}")

    t6 = torch.tensor([1, 2, 3])
    t7 = torch.tensor([4, 5, 6])
    t8 = t6.dot(t7)
    print(f"t8: {t8}")

if __name__ == '__main__':
    dm01()