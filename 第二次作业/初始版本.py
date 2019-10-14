# !/usr/bin/env python
# coding:utf-8
# Author: Lixiangyang
#将所有数据集用于训练,并且使用了relu函数，得到的准确率为100%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from sklearn import datasets#从sklearn中得到鸢尾花的数据集
import pandas as pd
iris_data=pd.read_csv('iris.txt',header=None)#读取数据
iris_data.replace("Iris-setosa",0,inplace = True)#相应的数据转为数字对应的类别
iris_data.replace("Iris-versicolor",1,inplace = True)
iris_data.replace("Iris-virginica",2,inplace = True)
data=np.array(iris_data)#转化为array类方便处理
x_data=np.array(data[:,[0,1,2,3]])
y_data=np.array(data[:,[-1]])
y_data=y_data.reshape(150)
y_data=y_data.astype(int)
#从自带的数据集中读取数据
#x_data = datasets.load_iris().data#x_data为鸢尾花的花萼长度，宽度，花瓣长度，宽度，共150组
#y_data = datasets.load_iris().target#y——data为鸢尾花的标签

#x_data = datasets.load_iris().data
np.random.seed(116)#使用seed对数据集进行扰乱
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
x_train=x_data#取出前150个数据集作为训练集
y_train=y_data
x_test=x_data[-30:]#取出后三十个作为测试集
y_test=y_data[-30:]

x_train = tf.cast(x_train, tf.float32)#在进行矩阵相乘时需要float型，故强制类型转换为float型
print(x_train.shape)
x_test=tf.cast(x_test,tf.float32)

# from_tensor_slices函数切分传入的 Tensor 的第一个维度，生成相应的 dataset
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)#训练时一次性读入32个训练集
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)#测试时一次性读入10个测试集
# iter用来生成迭代器
train_iter = iter(train_db)
# next() 返回迭代器的下一个项目
sample = next(train_iter)
w1 = tf.Variable(tf.random.truncated_normal([4, 32], stddev=0.1, seed=1))#w1为4行32列测参数,正态分布
b1 = tf.Variable(tf.random.truncated_normal([32], stddev=0.1, seed=1))#b1位32行，1列的参数，正态分布
w2 = tf.Variable(tf.random.truncated_normal([32, 32], stddev=0.1,seed=2))#w2为第二层参数
b2=tf.Variable(tf.random.truncated_normal([32], stddev=0.1,seed=2))
w2 = tf.Variable(tf.random.truncated_normal([32, 32], stddev=0.1,seed=2))#w2为第二层参数
b2=tf.Variable(tf.random.truncated_normal([32], stddev=0.1,seed=2))
w3 = tf.Variable(tf.random.truncated_normal([32, 3], stddev=0.1,seed=3))#w3为第三层参数
b3=tf.Variable(tf.random.truncated_normal([3], stddev=0.1,seed=3))
lr = 0.1#学习率
train_loss_results = []
epoch = 1000#迭代次数
loss_all=0#总的损失
for epoch in range(epoch):  #数据集级别迭代
    for step,(x_train,y_train) in enumerate(train_db):  #batch级别迭代
        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_train, w1) + b1
            h11=tf.nn.relu(h1)
            h2 = tf.matmul(h11, w2) + b2
            h22 = tf.nn.relu(h2)
            y = tf.matmul(h2,w3) + b3
            y_onehot = tf.one_hot(y_train, depth=3)#采用独热编码

            loss = tf.reduce_mean(tf.square(y_onehot - y))#采用平方差损失的平均值
            loss_all += loss.numpy()#将loss的张量转化为numpy型用来相加

            # compute gradients 进行梯度的计算
            grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])#分别计算w1,b1,w2,b2,w3,b3的梯度
            # w1 = w1 - lr * w1_grad 对梯度进行更新
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])
            w2.assign_sub(lr * grads[2])
            b2.assign_sub(lr * grads[3])
            w3.assign_sub(lr * grads[4])
            b3.assign_sub(lr * grads[5])

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))#每进行100步，输出一个loss
        train_loss_results.append(loss_all / 3)#将loss结果放入数组中
        loss_all = 0

    #在一个batch迭代完之后，用训练好的模型做测试
    total_correct, total_number = 0, 0
    for step, (x_train, y_train) in enumerate(test_db):
        h1 = tf.matmul(x_train, w1) + b1
        h11 = tf.nn.relu(h1)
        h2 = tf.matmul(h11, w2) + b2
        h22 = tf.nn.relu(h2)
        y = tf.matmul(h22, w3) + b3

        pred = tf.argmax(y, axis=1)#返回y每一行的最大值

        # 因为pred的dtype为int64，在计算correct时会出错，所以需要将它转化为int32
        pred = tf.cast(pred, dtype=tf.int32)#进行强制类型转换
        correct = tf.cast(tf.equal(pred, y_train), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_train.shape[0]
    acc = total_correct / total_number#每一个batch训练完后输出一个准确率
    print("test_acc:", acc)

# 绘制loss曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results)
plt.show()

