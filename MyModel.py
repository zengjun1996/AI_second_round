from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, Reshape, Conv2D, Add
from keras.layers import BatchNormalization, LeakyReLU, MaxPooling2D, Flatten
from keras.models import Model
from utils import *
import struct
import numpy as np
import time
import os
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# tf_config = tf.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# session = tf.Session(config=tf_config)

# mode = 0
SNRdb = 10
# Pilotnum = 32
# 以下仅为信道数据载入和链路使用范例

data1 = open('data/H.bin', 'rb')
H1 = struct.unpack('f'*2*2*2*32*320000, data1.read(4*2*2*2*32*320000))
H1 = np.reshape(H1, [320000, 2, 4, 32])
H = H1[:, 1, :, :]+1j*H1[:, 0, :, :]

y_val = pd.read_csv('data/y_val.csv', header=None)
x_val = pd.read_csv('data/x_val.csv', header=None)


# 使用链路和信道数据产生训练数据
def generator(batch, H):
    while True:
        input_labels = []
        input_samples = []
        for row in range(batch):
            bits0 = np.random.binomial(n=1, p=0.5, size=(128*4,))  # (512,)
            bits1 = np.random.binomial(n=1, p=0.5, size=(128*4,))  # (512,)
            X = [bits0, bits1]
            temp = np.random.randint(0, len(H))
            channel = H[temp]
            mode = np.random.randint(0, high=3, size=1)
            pilot_index = np.random.randint(0, high=2, size=1)
            if pilot_index == 0:
                Pilotnum = 8
            else:
                Pilotnum = 32
            y_data = MIMO(X, channel, SNRdb, mode, Pilotnum)/20
            x_data = np.concatenate((bits0, bits1), 0)
            input_labels.append(x_data)
            input_samples.append(y_data)
        batch_y = np.asarray(input_samples)
        batch_x = np.asarray(input_labels)
        yield batch_y, batch_x


data_input = Input(shape=(2048,))

x = Reshape((256, 4, 2))(data_input)

x = Conv2D(64, kernel_size=(3, 3), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*64

x = Conv2D(64, kernel_size=(3, 3), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*64

x = Conv2D(128, kernel_size=(3, 3), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*128

x = Conv2D(128, kernel_size=(3, 3), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*128

x = Conv2D(256, kernel_size=(3, 3), padding='SAME')(x)
x = BatchNormalization()(x)
x_ini = LeakyReLU()(x)  # 256*4*256

x = Conv2D(256, kernel_size=(3, 3), padding='SAME')(x_ini)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*256

x = Conv2D(256, kernel_size=(3, 3), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*256

x = Conv2D(256, kernel_size=(3, 3), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*256
x_ini = Add()([x_ini, x])

x = Conv2D(512, kernel_size=(3, 3), padding='SAME')(x_ini)
x = BatchNormalization()(x)
x_ini = LeakyReLU()(x)  # 256*4*512

x = Conv2D(512, kernel_size=(3, 3), padding='SAME')(x_ini)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*512

x = Conv2D(512, kernel_size=(3, 3), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*512

x = Conv2D(512, kernel_size=(3, 3), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*512
x_ini = Add()([x_ini, x])

x = Conv2D(512, kernel_size=(3, 3), padding='SAME')(x_ini)
x = BatchNormalization()(x)
x_ini = LeakyReLU()(x)  # 256*4*512

x = Conv2D(512, kernel_size=(3, 3), padding='SAME')(x_ini)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*512

x = Conv2D(512, kernel_size=(3, 3), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*512

x = Conv2D(512, kernel_size=(3, 3), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*512
x = Add()([x_ini, x])

x = Conv2D(2, kernel_size=(3, 3), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)  # 256*4*2

# x = MaxPooling2D(pool_size=(2, 2), padding='SAME')(x)
x = Flatten()(x)  # 2048
# x = Dropout(rate=0.5)(x)
data_output = Dense(1024, activation='sigmoid')(x)
model = Model(inputs=data_input, outputs=data_output)

print(model.summary())
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='mse', metrics=['binary_accuracy'])
tensorboard = TensorBoard(log_dir='logs/{}'.format(time.strftime('%m_%d_%H_%M')))
path = 'saved_models/model_{epoch:03d}.h5'
checkpoint = ModelCheckpoint(filepath=path, monitor='val_loss', period=100, verbose=1, save_best_only=True)
time_start = time.perf_counter()

model.fit_generator(generator(100, H),
                    steps_per_epoch=100,
                    epochs=2000,
                    validation_data=(y_val, x_val),
                    callbacks=[tensorboard, checkpoint])  # (10000,50,2000)

time_end = time.perf_counter()
print("训练耗时{:.2f}h".format((time_end-time_start)/3600))


# Y, X = generatorXY(10000, H)
# np.savetxt('Y_1.csv', Y, delimiter=',')
# X_1 = np.array(np.floor(X + 0.5), dtype=np.bool)
# X_1.tofile('X_1.bin')
