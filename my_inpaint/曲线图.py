import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv

'''读取csv文件'''


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2]))
        x.append((row[1]))
    return x, y


x, y = readcsv("smooth_4.csv")
x = list(map(int, x[1:]))
y = list(map(float, y[1:]))
plt.plot(x, y)
plt.ylim((-16, 4))
plt.xlim((0, 180000))

plt.xticks([1000 * i for i in range(0, 180)])  # 设置x轴刻度
plt.yticks([i * 2 for i in range(-14, 0)])  # 设置y轴刻度

plt.xlabel('Steps')
plt.ylabel('Score')
plt.show()
