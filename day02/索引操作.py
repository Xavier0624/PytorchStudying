"""
    张量的索引操作
    包括简单索引，列表索引，范围索引，布尔索引，多维索引
"""


import torch


torch.manual_seed(24)

t1 = torch.randint(0, 10, (5, 5))
print(f"t1: {t1}")

# 简单索引
print(f"t1[0]: {t1[0]}")
print(f"t1[0][0]: {t1[0][0]}")
print(f"t1[1]: {t1[:, 1]}")

# 列表索引
print(f"t1[[0, 2]]: {t1[[0, 2]]}")
print(t1[[0, 1], [1, 2]])


print(t1[[[0], [1]], [1, 2]])


# 范围索引
print(t1[:3, :2])
print(t1[2:,:2])
print(t1[1::2, ::2])

# 布尔索引
print(t1[torch.tensor([True, False, True, False, True]), :])

print(t1[t1[:, 2] > 5])
