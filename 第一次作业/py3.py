# !/usr/bin/env python
# coding:utf-8
# Author: Lixiangyang
from PIL import Image  # 引入PIL
import matplotlib.pyplot as plt
import numpy as np
image_path = "./mnist_data_jpg/"
f=open("./mnist.txt","r")
contents = f.readlines()
#print(contents)
for content in contents:
    value = content.split()
    img_path = image_path + value[0]
    img = Image.open(img_path)#img_path已经是字符串类型，就没有必要加字符串符号
    img = img.resize((28, 28), Image.ANTIALIAS)
    img = np.array(img.convert("L"))
    for i in range(28):
        for j in range(28):
            img[i][j]=255-img[i][j]# 将黑色区域变为白色，将白色变为黑色
    for i in range(28):   #对图片进行二值化
        for j in range(28):
            if(img[i][j]>230):
                img[i][j]==255
            else:
                img[i][j]==0
    im = Image.fromarray(img)  # 将array转成Image
    im.save(img_path)  # 保存图片
