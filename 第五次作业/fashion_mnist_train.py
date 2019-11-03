#author:lixiangyang
import tensorflow as tf
from PIL import Image
import numpy as np

np.set_printoptions(threshold=float('inf'))
model_save_path = './fashion_mnist_checkpoint/mnist.tf'
load_pretrain_model = True

train_path='./fashion_mnist_train_jpg_60000/'
train_txt='./fashion_mnist_train_jpg_60000.txt'

test_path='./fashion_mnist_test_jpg_10000/'
test_txt='./fashion_mnist_test_jpg_10000.txt'

def generateds(path,txt):
    f = open(txt, 'r')
    contents=f.readlines()#按行读取
    f.close()
    images, labels = [], []
    for content in contents:
        value=content.split()  #以空格分开，存入数组
        img_path=path+value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
        img=img/255.
        images.append(img)
        labels.append(value[1])
        print('loading : '+content)

    x_=np.array(images)
    y_=np.array(labels)
    y_ = y_.astype(np.int64)
    return x_,y_

print('-------------load the data-----------------')
x_train,y_train=generateds(train_path,train_txt)
x_test,y_test=generateds(test_path,test_txt)

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')
                                    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

if load_pretrain_model:
    print ('-------------load the model-----------------')
    model.load_weights(model_save_path)
model.summary()

for i in range(5):
    model.fit(x_train, y_train, epochs=10,batch_size=512,validation_data=(x_test,y_test),validation_freq=2)
    model.save_weights(model_save_path, save_format='tf')

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(v.name +'\n')
    file.write(str(v.numpy()) +'\n')
file.close()
