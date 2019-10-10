# !/usr/bin/env python
# coding:utf-8
# Author: Lixiangyang
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("SH_600519_high_low.csv")
x = np.arange(0, 47)
y1 = df["high"]
y2 = df["low"]
print("x:\n", x, "\ny1:\n", y1, "\ny2:\n", y2)
plt.xlabel("day")
plt.ylabel("price")
plt.title("Kweichow Moutai")
plt.subplot(1, 2, 1) #1行2列的第一个
plt.plot(x, y1, "b",label="high",)
plt.legend()#显示图例
plt.subplot(1, 2, 2) #1行2列的第一个
plt.plot(x, y2, "yellow",label="low")
plt.legend()#显示图例
plt.show()
