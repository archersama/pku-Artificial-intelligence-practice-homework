#author:lixiangyang
# 使用前
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Flatten,Dense
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_path='./train/'
test_path='./test/'
model_save_path = './'

def generateds(path):
    imagelist = os.listdir(path)  # 读取images文件夹下所有文件的名字
    images, labels = [], []
    for label,content in enumerate(imagelist):
        img_path=path+content
        imags=os.listdir(img_path)
        for j in range(len(imags)):
            img=Image.open(img_path+"/"+imags[j])
            img=np.array(img)
            images.append(img)
            labels.append(label)
            print('loading : '+img_path+"/"+imags[j])
    x_=np.array(images)
    y_=np.array(labels)
    y_ = y_.astype(np.int64)
    return x_,y_
x_train,y_train=generateds(train_path)
x_test,y_test=generateds(test_path)

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_train, x_test = x_train / 255.0, x_test / 255.0

print(y_train[0])

#对数据集进行打乱
np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
print(y_train[0])


image_gen_train = ImageDataGenerator(

                                     rotation_range=10,#随机45度旋转
                                     width_shift_range=0.15,#宽度偏移
                                     height_shift_range=0.15,#高度偏移
                                     horizontal_flip=True,#水平翻转
                                     zoom_range=0.8#将图像随机缩放到50％
                                     )
image_gen_train.fit(x_train)

model = tf.keras.models.Sequential([
    Conv2D(filters=64,kernel_size=(5,5),padding='same',input_shape=(32,32,3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Dropout(0.2),


    Conv2D(64, kernel_size=(5,5), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
    Dropout(0.2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

if os.path.exists(model_save_path+'first_model.h5'):
    print('-------------load the model-----------------')
    model=tf.keras.models.load_model('first_model.h5')
for i in range(5):
    history=model.fit(image_gen_train.flow(x_train, y_train,batch_size=32), epochs=20, validation_data=(x_test, y_test), validation_freq=1)
    model.save(os.path.join(os.path.dirname('./'), 'first_model.h5'))

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
