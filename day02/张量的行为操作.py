"""
    reshape(a, newshape, order='C')
    squeeze(a, axis=None)
    unsqueeze(a, axis)
    transpose(a, axes=None)
    permute(a, *dims)
    view(a, dtype=None, type=None)
    contiguous(a)
"""


import torch

torch.initial_seed()


def dm01():


    t1 = torch.randint(1, 10, (2, 3))
    print(f"t1: {t1}, shape: {t1.shape}, row: {t1.shape[0]}, col: {t1.shape[1], t1.shape[-1]}")

    t2 = t1.reshape(3, 2)
    print(f"t2: {t2}, shape: {t2.shape}, row: {t2.shape[0]}, col: {t2.shape[1], t2.shape[-1]}")

    t3 = t1.reshape(1, 6)
    print(f"t3: {t3}, shape: {t3.shape}, row: {t3.shape[0]}, col: {t3.shape[1], t3.shape[-1]}")

def dm02():
    t1 = torch.randint(1, 10, (2, 3))
    print(f"t1: {t1}, shape: {t1.shape}")
    t2 = t1.unsqueeze(0)
    print(f"t1: {t2}, shape: {t2.shape}")

    t3 = t1.unsqueeze(1)
    print(f"t1: {t3}, shape: {t3.shape}")

    t4 = t1.unsqueeze(2)
    print(f"t1: {t4}, shape: {t4.shape}")

    t6 = torch.randint(1, 10, (2, 1, 3, 1, 1))
    print(f"t6: {t6}, shape: {t6.shape}")

    t7 = t6.squeeze()
    print(f"t7: {t7}, shape: {t7.shape}")


def dm03():
    t1 = torch.randint(1, 10, (2, 3, 4))
    print(f"t1: {t1}, shape: {t1.shape}")
    t2 = t1.transpose(0, -1)
    print(f"t2: {t2}, shape: {t2.shape}")
    t3 = t1.permute(1, 0, 2)
    print(f"t3: {t3}, shape: {t3.shape}")


def dm04():
    t1 = torch.randint(1, 10, (2, 3))
    print(f"t1: {t1}, shape: {t1.shape}")

    print(f"{t1.is_contiguous()}")
    t2 = t1.view(3, 2)
    print(f"t2: {t2}, shape: {t2.shape}")
    print(t2.is_contiguous())

    t3 = t1.transpose(0, 1)
    print(t3.is_contiguous())

    t4 = t3.contiguous().view(2, 3)
    print(f"t4: {t4}, shape: {t4.shape}")


if __name__ == '__main__':
    # dm02()
    # dm01()
    # dm03()
    dm04()