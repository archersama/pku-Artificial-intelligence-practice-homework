# coding:utf-8
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,LSTM,GRU
from tensorflow.keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

maotai = pd.read_csv('./SH600519.csv')

training_set = maotai.iloc[0:2186 - 300, 1:2].values
test_set = maotai.iloc[2186 - 300:, 1:2].values

# 归一化
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set = sc.transform(test_set)

x_train = []
y_train = []

x_test = []
y_test = []

#                     2186-300
# 前60天的数据当做输入x,第61天数据当作目标y
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# print(x_train.shape)
# print(y_train.shape)
# print('x_train:', x_train)
# print('x_train:', x_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

for i in range(60, len(test_set)):
    x_test.append(test_set[i - 60:i, 0])
    y_test.append(test_set[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


print(x_train.shape)
print(y_train.shape)
print('x_train:', x_train)
print('x_train:', x_train.shape)

model = tf.keras.Sequential([
    # tf.keras.layers.Embedding(5, 1,input_length=1),
    #SimpleRNN(50, return_sequences=True, unroll=True),
    # Dropout(0.2),
    GRU(128, activation='relu',return_sequences=False, unroll=True),
    Dropout(0.2),
    # SimpleRNN(128, return_sequences=True, unroll=False),
    # Dropout(0.2),
    # LSTM(50,activation='relu',return_sequences=False,unroll=True),
    # Dropout(0.2),
    Dense(1)
])
sgd=tf.keras.optimizers.SGD(0.001)
adam=tf.keras.optimizers.Adam()
model.compile(optimizer=sgd,
              loss='mean_squared_error')

checkpoint_save_path = "./checkpoint3/stock.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 # monitor='loss',
                                                 save_best_only=True,
                                                 verbose=1)

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback], verbose=1)

model.summary()

file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

################## predict ######################

predicted_stock_price = model.predict(x_test)
# print(predicted_stock_price)
# 对预测数据还原。
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# print(predicted_stock_price)
# print(test_set[60:])

real_stock_price = sc.inverse_transform(test_set[60:])

plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price')
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MaoTai Stock Price')
plt.legend()
plt.show()