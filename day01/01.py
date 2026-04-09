import numpy as np
import torch


def dm01():
    t1 = torch.zeros(2, 3)
    t2 = torch.ones(2, 3)
    t3 = torch.full(size=(2, 3), fill_value=100)
    t4 = torch.ones_like(t1)
    t5 = torch.zeros_like(t2)
    t6 = torch.full_like(t3, fill_value=200)



    print(f't1: {t1}, type: {type(t1)}')
    print(f't2: {t2}, type: {type(t2)}')
    print(f't3: {t3}, type: {type(t3)}')
    print(f't4: {t4}, type: {type(t4)}')
    print(f't5: {t5}, type: {type(t5)}')
    print(f't6: {t6}, type: {type(t6)}')




if __name__ == '__main__':
    dm01()
