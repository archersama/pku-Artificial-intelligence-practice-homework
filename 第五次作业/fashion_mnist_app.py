#author:lixiangyang
#10张图片错了4个，分别为coat，sandal，shirt，sneaker，如果有更好的图片处理方法，欢迎提出issue以及pull request
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.set_printoptions(threshold=float('inf'))
model_save_path = './fashion_mnist_checkpoint/mnist.tf'
def print_name(pred):
    if(pred==0):
        print("t-shirt")
    elif(pred==1):
        print("trouser")
    elif(pred==2):
        print("coat")
    elif(pred==3):
        print("dress")
    elif(pred==4):
        print("pullover")
    elif(pred==5):
        print("sandal")
    elif(pred==6):
        print("shirt")
    elif(pred==7):
        print("sneaker")
    elif(pred==8):
        print("bag")
    elif(pred==9):
        print("ankle_boot")


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(model_save_path)

preNum = int(input("input the number of test pictures:"))
for i in range(preNum):
    image_path = input("the path of test picture:")
    img = Image.open(image_path)

    imge = plt.imread(image_path)
    plt.imshow(imge, cmap=plt.cm.binary)
    plt.show()

    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    plt.imshow(img_arr, cmap=plt.cm.binary)
    plt.show()
    for i in range(28):
        for j in range(28):
            if img_arr[i][j]<180:
                img_arr[i][j]=255
            else:
                img_arr[i][j]=0
    plt.imshow(img_arr, cmap=plt.cm.binary)
    plt.show()
    img_arr = img_arr / 255.0
    plt.imshow(img_arr, cmap=plt.cm.binary)
    plt.show()
    x_predict = img_arr[tf.newaxis, ...]
    #print(x_predict)
    result = model.predict(x_predict)
    #print(result)
    pred = tf.argmax(result, axis=1)
    print('\n')
    tf.print(pred)
    print_name(pred)
    plt.pause(10)
    plt.close()


