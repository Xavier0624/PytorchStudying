"""
 cat(): 将多个张量连接在一起，形成一个新的张量。它可以沿着指定的维度进行连接。
 stack(): 改变维度数量，将多个张量堆叠在一起，形成一个新的张量。它会在指定的维度上增加一个新的维度。
"""


import torch

torch.initial_seed()

t1 = torch.randint(1, 10, (2, 3))
t2 = torch.randint(1, 10, (2, 3))

print(f"t1: {t1}, shape: {t1.shape}")
print(f"t2: {t2}, shape: {t2.shape}")


t3 = torch.cat((t1, t2), dim=0)
print(f"t3: {t3}, shape: {t3.shape}")


t4 = torch.stack((t1, t2), dim=0)
print(f"t4: {t4}, shape: {t4.shape}")

t5 = torch.stack((t1, t2), dim=1)
print(f"t5: {t5}, shape: {t5.shape}")

t6 = torch.stack((t1, t2), dim=-1)
print(f"t6: {t6}, shape: {t6.shape}")