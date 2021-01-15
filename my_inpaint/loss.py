# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:XLin
# datetime:2020/4/2 0002 18:13
# software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import random

plt.plot([i for i in range(500000)], [random.uniform(-5, 5) for _ in range(500000)])

# plt.xlim(0, 500000)
# plt.ylim(0, 15)

plt.title('line chart')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
