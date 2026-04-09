"""
参数初始化的七种方式：
uniform_
normal_
constant_
zeros_
ones_
xavier_uniform_
xavier_normal_
kaiming_normal_
kaiming_uniform_
"""

import  torch.nn as nn


def dm01():
    # 创建一个线性层
    linear = nn.Linear(5, 3)
    # 使用uniform_方法进行参数初始化
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print(linear.weight.data)
    print(linear.bias)

def dm02():
    linear = nn.Linear(5, 3)
    # nn.init.normal_(linear.weight)
    # nn.init.normal_(linear.bias)
    # nn.init.zeros_(linear.weight)
    # nn.init.zeros_(linear.bias)
    nn.init.kaiming_normal_(linear.weight)
    print(linear.weight.data)

if __name__ == '__main__':
    dm02()