"""
    detach()

"""

import torch
import numpy as np


t1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float32)

t2 = t1.detach()  # t2与t1共享内存，但t2不参与计算图的构建
print(f"t1: {t1}, t2: {t2}")


t1.data[0] = 100
print(f"t1: {t1}, t2: {t2}")  # t2的值也发生了改变，因为t1和t2共享内存

n1 = t1.detach().numpy()  # n1与t1共享内存，但n1不参与计算图的构建
print(f"n1: {n1}, type: {type(n1)}")