# 利用sequential结构训练并测试cifar10
#author:lixiangyang
#采用VGG19net结构，在测试集上准确率可达92%以上
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Flatten,Dense
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator



cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_train, x_test = x_train / 255.0, x_test / 255.0


image_gen_train = ImageDataGenerator(
                                     rotation_range=10,#随机10度旋转
                                     width_shift_range=0.1,#宽度偏移
                                     height_shift_range=0.1,#高度偏移
                                     horizontal_flip=True,#水平翻转
                                     )
image_gen_train.fit(x_train)


model = tf.keras.models.Sequential([
    Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',input_shape=(32,32,3)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',input_shape=(32,32,3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2),strides=2,padding='same'),
    Dropout(0.2),

    Conv2D(128, kernel_size=(3,3),strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, kernel_size=(3,3),strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
    Dropout(0.2),

    Conv2D(256, kernel_size=(3,3),strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel_size=(3,3),strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel_size=(3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel_size=(3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
    Dropout(0.2),

    Conv2D(512, kernel_size=(3,3),strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel_size=(3,3),strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel_size=(3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel_size=(3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
    Dropout(0.2),

    Conv2D(512, kernel_size=(3,3),strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel_size=(3,3),strides=1,padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel_size=(3,3),strides=1,padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel_size=(3,3),strides=1,padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
    Dropout(0.2),


    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
checkpoint_save_path = "./checkpoint/cifar10.ckpt"
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 # monitor='loss',
                                                 save_best_only=True,
                                                 verbose=2)
history = model.fit(image_gen_train.flow(x_train, y_train,batch_size=16),epochs=30,
                    validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback], verbose=1)

model.summary()
#保存模型参数到txt文件中
file=open('./weights.txt','w')
for value in model.trainable_variables:
    file.write(str(value.name)+'\n')
    file.write(str(value.shape) + '\n')
    file.write(str(value.numpy()) + '\n')

model.summary()
# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
