#author:lixiangyang
#在运行断点续训.py前请先运行第模型训练.py来得到断点
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

#加载已经保存好的模型继续训练
restored_model = tf.keras.models.load_model('first_model.h5')
restored_model.compile(optimizer=tf.keras.optimizers.Adam(),
             loss = 'sparse_categorical_crossentropy',
             metrics=['accuracy'])
restored_model.fit(x_train,y_train,epochs=10,batch_size=512,validation_data=(x_test,y_test),validation_freq=2)
#将训练好的模型进行保存
restored_model.save('first_model.h5')
