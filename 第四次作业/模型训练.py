#author：lixiangyang
#采用五层全连接神经网络，每层使用0.1概率的dropout，200次迭代训练集准确率可达97%，增加迭代次数会再次上升
# 测试集准确率稳定在89%左右，如果有更好的全连接神经网络模型，欢迎提出issue或直接pull request
import tensorflow as tf
import tensorflow.keras
import os
import matplotlib.pyplot as plt
import numpy as np
import logging


from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#显示y_train[0] (文本) 和 x_train[0] (文本和图片)
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
print(y_train[0])
x_train = x_train / 255.0
x_test = x_test / 255.0


from tensorflow import keras


model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(128, activation='relu',),
                                    keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(32, activation='relu'),
                                    keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(16, activation='relu'),
                                    keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(10, activation='softmax')
                                    ])

model.compile(optimizer=tf.keras.optimizers.Adam(),
             loss = 'sparse_categorical_crossentropy',
             metrics=['accuracy'])

#保存断点，保存数据历史
try:
    history=model.fit(x_train, y_train, epochs=200,
              batch_size=512,
              validation_data=(x_test, y_test),
              validation_freq=2)
except KeyboardInterrupt:
    #如果被打断则对模型进行保存
    model.save_weights('./'.format(epoch=0))
    logging.info('keras model saved.')
model.save_weights('./'.format(epoch=0))

#将模型保存为h5文件格式，方便载入
model.save(os.path.join(os.path.dirname('./'), 'first_model.h5'))
model.summary()


#查看保存的历史数据格式
print(history.history.keys())
#画出准确率曲线
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#画出损失曲线
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
