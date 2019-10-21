#author:lixiangyang
import tensorflow as tf
import os
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息


def normalize(train):
    # 线性归一化
    x_data = train.T
    for i in range(4):
        x_data[i] = (x_data[i] - tf.reduce_min(x_data[i])) / (tf.reduce_max(x_data[i]) - tf.reduce_min(x_data[i]))
    return x_data.T


def norm_nonlinear(train):
    # 非线性归一化（log）
    x_data = train.T
    for i in range(4):
        x_data[i] = np.log10(x_data[i]) / np.log10(tf.reduce_max(x_data[i]))
    return x_data.T


def standardize(train):
    # 数据标准化（标准正态分布）
    x_data = train.T
    for i in range(4):
        x_data[i] = (x_data[i] - np.mean(x_data[i])) / np.std(x_data[i])
    return x_data.T


x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

x_data = standardize(x_data)

# 随机打乱数据
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

print("x.shape:", x_data.shape)
print("y.shape:", y_data.shape)
print("x.dtype:", x_data.dtype)
print("y.dtype:", y_data.dtype)
print("min of x:", tf.reduce_min(x_data))
print("max of x:", tf.reduce_max(x_data))
print("min of y:", tf.reduce_min(y_data))
print("max of y:", tf.reduce_max(y_data))

# from_tensor_slices函数切分传入的 Tensor 的第一个维度，生成相应的 dataset
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)

# iter用来生成迭代器
train_iter = iter(train_db)
# next() 返回迭代器的下一个项目
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

w1 = tf.Variable(tf.random.truncated_normal([4, 32], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([32], stddev=0.1, seed=11))
w2 = tf.Variable(tf.random.truncated_normal([32, 3], stddev=0.1, seed=4))
b2 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=44))

learning_rate_step = 10
learning_rate_decay = 0.8
train_loss_results = []
test_acc = []
lr = []
global_step=epoch = 500
loss_all = 0
learning_rate_base=0.01
m_w1, m_b1 = 0, 0
m_w2, m_b2 = 0, 0
v_w1, v_b1 = 0, 0
v_w2, v_b2 = 0, 0
beta1, beta2 = 0.9, 0.999
train_mode = True
train_accurate_results = []
if train_mode:
    for epoch in range(epoch):
        learning_rate = learning_rate_base * tf.sqrt(1-beta2)/(1-beta1)
        lr.append(learning_rate)
        for step, (x_train, y_train) in enumerate(train_db):  # enmuerate 可以遍历数据对象
            with tf.GradientTape() as tape:  # with 记录计算过程
                h1 = tf.matmul(x_train, w1) + b1  # x_train [32,4] * w1 [4,32] = h1 [32,32]
             # 同向传播  relu发射
                h11 = tf.nn.relu(h1)
                y = tf.matmul(h11, w2) + b2
                y=tf.nn.sigmoid(y)
                y_one_hot = tf.one_hot(y_train, depth=3)  # 将 input转为one-hot输出
                # tf.one_hot（待转换数据，depth = 几分类）
                # mse = mean(sum(y-out)^2)
                loss = tf.reduce_mean(tf.square(y_one_hot - y))
                #交叉熵损失函数，使用时将上面的一个损失函数注释掉
                #loss=tf.reduce_mean(tf.losses.categorical_crossentropy(y_one_hot,y))
                loss_all += loss.numpy()

                # compute gradients
                grads = tape.gradient(loss, [w1, b1, w2, b2])  # 记录计算梯度  loss对w1,b1,w2...的梯度
                # w1 = w1 - lr * w1_grad                                        # 反向传播
                # adam
                m_w1 = beta1 * m_w1 + (1 - beta1) * grads[0]
                m_b1 = beta1 * m_b1 + (1 - beta1) * grads[1]
                v_w1 = beta2 * v_w1 + (1 - beta2) * tf.square(grads[0])
                v_b1 = beta2 * v_b1 + (1 - beta2) * tf.square(grads[1])
                m_w2 = beta1 * m_w2 + (1 - beta1) * grads[2]
                m_b2 = beta1 * m_b2 + (1 - beta1) * grads[3]
                v_w2 = beta2 * v_w2 + (1 - beta2) * tf.square(grads[2])
                v_b2 = beta2 * v_b2 + (1 - beta2) * tf.square(grads[3])
                m_w1_correction = m_w1 / (1 - tf.pow(beta1, int(global_step)))
                m_b1_correction = m_b1 / (1 - tf.pow(beta1, int(global_step)))
                v_w1_correction = v_w1 / (1 - tf.pow(beta2, int(global_step)))
                v_b1_correction = v_b1 / (1 - tf.pow(beta2, int(global_step)))
                m_w2_correction = m_w2 / (1 - tf.pow(beta1, int(global_step)))
                m_b2_correction = m_b2 / (1 - tf.pow(beta1, int(global_step)))
                v_w2_correction = v_w2 / (1 - tf.pow(beta2, int(global_step)))
                v_b2_correction = v_b2 / (1 - tf.pow(beta2, int(global_step)))
                w1.assign_sub(learning_rate * m_w1_correction / tf.sqrt(v_w1_correction))
                b1.assign_sub(learning_rate * m_b1_correction / tf.sqrt(v_b1_correction))
                w2.assign_sub(learning_rate * m_w2_correction / tf.sqrt(v_w2_correction))
                b2.assign_sub(learning_rate * m_b2_correction / tf.sqrt(v_b2_correction))
            if step % 10 == 0:
                    print(epoch, step, 'loss:', float(loss))
        train_loss_results.append(loss_all)
        loss_all = 0

        # test model(做测试）

        total_correct, total_number = 0, 0
        for step, (x_train, y_train) in enumerate(test_db):
            h1 = tf.matmul(x_train, w1) + b1  # x_train [32,4] * w1 [4,32] = h1 [32,32]
            # 同向传播  relu发射
            h11 = tf.nn.relu(h1)
            y = tf.matmul(h11, w2) + b2
            y=tf.nn.sigmoid(y)
            pred = tf.argmax(y, axis=1)
            # 因为pred的dtype为int64，在计算correct时会出错，所以需要将它转化为int32
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y_train), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_correct += int(correct)
            total_number += x_train.shape[0]
        test_acc = total_correct / total_number
        print("total_correct", total_correct, "total_number", total_number)
        print("test_acc:", test_acc)
        train_accurate_results.append(test_acc)
        test_acc = 0

plt.subplot(311)
# 绘制 loss 曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label="$ adam-mse-Loss $",color='blue')
plt.legend()
# 绘制 Accuracy 曲线
plt.subplot(312)
plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(train_accurate_results, label="$adam-mse-Accuracy$",color="blue")
plt.legend()
plt.subplot(313)
# 绘制 Learning_rate 曲线
plt.title('Learning Rate Curve')
plt.xlabel('Global steps')
plt.ylabel('Learning rate')
plt.plot(range(global_step), lr, label="$adam-mse-lr$",color="blue")
plt.legend()
plt.show()
