

import torch
import torch.nn as nn

def dm01():
    t1 = torch.randint(0, 10, (1, 4)).float()

    # print(t1)

    linear1 = nn.Linear(4, 5)
    l1 = linear1(t1)
    print(f"l1: {l1}")
    output = torch.relu(l1)
    print(f"output: {output}")


    dropout = nn.Dropout(p=0.4)
    output_dropout = dropout(output)
    print(f"output_dropout: {output_dropout}")

def dm02():
    t1 = torch.randn(size=(1,2,3,4)) # 一张图片，两个通道，每个通道3行4列
    print(f"t1: {t1}")

    # 参1:输入特征数等于通道数，参2:小常数，参3:动量，参4:是否学习
    bn2d = nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True,)

    op2d = bn2d(t1)
    print(f"op2d: {op2d}")

if __name__ == '__main__':
    dm02()