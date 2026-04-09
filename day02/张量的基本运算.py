"""
    add
    sub
    mul
    div
    neg
    add_
    sub_
    mul_
    div_
    neg_
"""


import torch

t1 = torch.tensor([1, 2, 3])

t2 = t1.add(10)

t2 = t1 + 10

t2 = t1.add_(10)

# t2 += 10
# t2 -= 10

print(f"t1: {t1}, t2: {t2}")