import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/My Drive"

os.chdir(path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

image_gen_train = ImageDataGenerator(
                                     rescale=1,#归至0～1
                                     rotation_range=0,#随机0度旋转
                                     width_shift_range=0,#宽度偏移
                                     height_shift_range=0,#高度偏移
                                     horizontal_flip=True,#水平翻转
                                     zoom_range=1#将图像随机缩放到100％
                                     )
image_gen_train.fit(x_train)


class ResNetBlk(Model):
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResNetBlk,self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path
        self.bn1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.conv1 = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same',
                         use_bias=False, kernel_initializer=tf.random_normal_initializer())  # 卷积层

        self.bn2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.conv2 = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same',
                         use_bias=False, kernel_initializer=tf.random_normal_initializer())  # 卷积层
        if residual_path:
            self.down_conv = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same',
                                use_bias=False, kernel_initializer=tf.random_normal_initializer())  # 卷积层
            self.down_bn = BatchNormalization()

    def call(self, x,training=None):
        x1 = self.bn1(x, training=training)
        x2 = self.a1(x1)
        x3 = self.conv1(x2, training=training)
        x4 = self.bn2(x3, training=training)
        x5 = self.a2(x4)
        x6 = self.conv2(x5, training=training)
        if self.residual_path:
            x7 = self.down_conv(x)
            x8 = self.down_bn(x7)
            x6 = tf.add(x6,x8)
        else:
            x6 = tf.add(x,x6)
        return x6

class Cifar10_Model(Model):
    def __init__(self, num_classes, **kwargs):
        super(Cifar10_Model, self).__init__(**kwargs)
        self.c1 = tf.keras.models.Sequential([Conv2D(64, 3, strides=1, padding='same')
                                                 ])
        self.blocks = tf.keras.models.Sequential(name='dynamic-blocks')
        for block_id in range(4):
            for layer_id in range(2):
                if layer_id or block_id == 0:
                    block = ResNetBlk(64*(block_id+1))
                else:
                    block = ResNetBlk(64*(block_id+1),strides=2,residual_path=True)
                self.blocks.add(block)
        self.bn = BatchNormalization()
        self.a = Activation('relu')
        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, x, training=None):
        x=self.c1(x,training=training)
        x=self.blocks(x,training=training)
        x=self.bn(x)
        x=self.a(x)
        x=self.p1(x)
        y=self.f1(x)
        return y

model = Cifar10_Model(10)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint_v2/cifar10-ResNet18.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 # monitor='loss',
                                                 # save_best_only=True,
                                                 verbose=2)
history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=128), epochs=200,
                    validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback], verbose=1, shuffle=True)

model.summary()

file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

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
