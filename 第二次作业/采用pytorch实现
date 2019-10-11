#采用pytorch实现，正确率为98%
import numpy as np
from collections import Counter
from sklearn import datasets
import torch.nn.functional as Fun
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch

#数据准备
dataset = datasets.load_iris()
dataut=dataset['data']
priciple=dataset['target']
#priciple=priciple.reshape([150,1])
#new_data=np.hstack((dataut,priciple))
input=torch.FloatTensor(dataset['data'])
label=torch.LongTensor(dataset['target'])
epoch=500

#定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = Fun.relu(self.hidden(x))      # activation function for hidden layer we choose sigmoid
        x = self.out(x)
        return x

net = Net(n_feature=4, n_hidden=20, n_output=3)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02) #SGD: 随机梯度下降
loss_func = torch.nn.CrossEntropyLoss() #针对分类问题的损失函数![在这里插入图片描述](https://img-blog.csdnimg.cn/20190108120127973.png)

#训练数据
for t in range(epoch):
        out = net(input)                 # input x and predict based on x
        loss = loss_func(out, label)     # 输出与label对比
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        if (t + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(t + 1, epoch, loss.item()))

for t in range(epoch):
    out = net(input)  # out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
    prediction = torch.max(out, 1)[1]  # 1返回index  0返回原值
    pred_y = prediction.data.numpy()
    target_y = label.data.numpy()
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    if (t + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(t + 1, epoch, loss.item()))

out = net(input) #out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
prediction = torch.max(out, 1)[1] # 1返回index  0返回原值
pred_y = prediction.data.numpy()
target_y = label.data.numpy()
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print("莺尾花预测准确率",accuracy)
