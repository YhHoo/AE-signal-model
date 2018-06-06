import numpy as np
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from dataset_experiment_16_5_2018 import AccousticEmissionDataSet_16_5_2018
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import optimizers
from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import correlate

# self library
# from ideal_dataset import noise_time_shift_dataset
#
# data_3d = np.array([[[1, 2],
#                   [3, 4]],
#                  [[2, 3],
#                   [4, 5]],
#                  [[3, 4],
#                   [5, 6]]])
# data_2d = np.array([[1, 2],
#                     [3, 4],
#                     [2, 5]])
# data_2d = data_2d.astype(dtype='float32')

# mat1 = np.array([1, 0, 3, 3, 2, 1, 0, 0]).reshape((1, 8))
# mat2 = np.array([3, 1, 0, 3, 3, 2, 1, 0]).reshape((1, 8))
#
# mat3 = np.array([0, 0, 1, 0, 1, 0, 0, 0]).reshape((1, 8))
# mat4 = np.array([0, 0, 0, 1, 0, 1, 0, 0]).reshape((1, 8))
#
# ori_signal = np.concatenate((mat1, mat3), axis=0)
# lag_signal = np.concatenate((mat2, mat4), axis=0)
# print(ori_signal.shape)

# cor = np.correlate(mat1, mat2, 'full')
# print('USING NUMPY CORRELATE---------------')
# print(cor)
# print(cor.shape)
# print(np.argmax(cor))

# cor = correlate(ori_signal, lag_signal, 'full')
# print('USING SCIPY CORRELATE---------------')
# print(cor)
# print(cor.shape)
# print(np.argmax(cor))


for i in range(10):
    lx = 4 + 5

print(lx)