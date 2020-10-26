from keras.optimizers import Adam
from keras.layers import Input, add, Dense
from keras.models import Model
from utils import *
import struct
import numpy as np
import time

mode = 0
SNRdb = 10
Pilotnum = 32
# 以下仅为信道数据载入和链路使用范例

data1 = open('data/H.bin', 'rb')
H1 = struct.unpack('f'*2*2*2*32*320000, data1.read(4*2*2*2*32*320000))
H1 = np.reshape(H1, [320000, 2, 4, 32])
H = H1[:, 1, :, :]+1j*H1[:, 0, :, :]

Htest = H[300000:, :, :]
H = H[:300000, :, :]

# 使用链路和信道数据产生训练数据
def generator(batch, H):
    while True:
        input_labels = []
        input_samples = []
        for row in range(0, batch):
            bits0 = np.random.binomial(n=1, p=0.5, size=(128*4,))  # (512,)
            bits1 = np.random.binomial(n=1, p=0.5, size=(128*4,))  # (512,)
            X = [bits0, bits1]
            temp = np.random.randint(0, len(H))
            HH = H[temp]
            YY = MIMO(X, HH, SNRdb, mode, Pilotnum)/20
            XX = np.concatenate((bits0, bits1), 0)
            input_labels.append(XX)
            input_samples.append(YY)
        batch_y = np.asarray(input_samples)
        batch_x = np.asarray(input_labels)
        yield batch_y, batch_x


time_start = time.perf_counter()
data_input = Input(shape=(2048,))
x = Dense(2048, activation='relu')(data_input)
data_output = Dense(1024, activation='sigmoid')(x)
model = Model(inputs=data_input, outputs=data_output)
print(model.summary())
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='mse', metrics=['binary_accuracy'])
model.fit_generator(generator(10, H), steps_per_epoch=50, epochs=10)  # (10000,50,2000)
model.save('model.h5')
time_end = time.perf_counter()
print("训练耗时{:.2f}".format(time_end-time_start))


# 产生测评数据，仅供参考格式
def generatorXY(batch, H):
    input_labels = []
    input_samples = []
    for row in range(0, batch):
        bits0 = np.random.binomial(n=1, p=0.5, size=(128*4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128*4,))
        X = [bits0, bits1]
        temp = np.random.randint(0, len(H))
        HH = H[temp]
        YY = MIMO(X, HH, SNRdb, mode, Pilotnum)/20
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    return batch_y, batch_x


# Y, X = generatorXY(10000, H)
# np.savetxt('Y_1.csv', Y, delimiter=',')
# X_1 = np.array(np.floor(X + 0.5), dtype=np.bool)
# X_1.tofile('X_1.bin')

y_test, x_test = generatorXY(10, H)  # 10000
print(model.evaluate(y_test, x_test))
