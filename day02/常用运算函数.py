"""
    sum,max,min,mean, 有dim参数可以指定行列
    pow,sqrt,exp,log,log2,log10
"""


import torch


t1 = torch.tensor([[1, 2, 3, 4, 6], [2, 2, 4, 2 ,1]], dtype=torch.float32)

print(f"t1: {t1}")

print(f"sum: {t1.sum()}")
print(f"sum dim=0: {t1.sum(dim=0)}")
print(f"sum dim=1: {t1.sum(dim=1)}")

print(f"max: {t1.max()}")
print(f"max dim=0: {t1.max(dim=0)}")
print(f"max dim=1: {t1.max(dim=1)}")

print(f"min: {t1.min()}")
print(f"min dim=0: {t1.min(dim=0)}")
print(f"min dim=1: {t1.min(dim=1)}")

print(f"mean: {t1.mean()}")
print(f"mean dim=0: {t1.mean(dim=0)}")
print(f"mean dim=1: {t1.mean(dim=1)}")


print(f"pow: {t1.pow(2)}")
print(f"sqrt: {t1.sqrt()}")
print(f"exp: {t1.exp()}")
print(f"log: {t1.log()}")
print(f"log2: {t1.log2()}")
print(f"log10: {t1.log10()}")
