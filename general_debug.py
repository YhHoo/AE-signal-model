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
# self library
# from ideal_dataset import noise_time_shift_dataset

data_3d = np.array([[[1, 2],
                  [3, 4]],
                 [[2, 3],
                  [4, 5]],
                 [[3, 4],
                  [5, 6]]])
data_2d = np.array([[1, 2],
                    [3, 4],
                    [2, 5]])
data_2d = data_2d.astype(dtype='float32')

scaler = MinMaxScaler(feature_range=(0, 1))

data_2d = scaler.fit_transform(data_2d.ravel().reshape(-1, 1)).reshape((data_2d.shape[0], data_2d.shape[1]))
print(data_2d)


