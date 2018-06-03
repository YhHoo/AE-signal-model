import numpy as np
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from dataset_experiment_16_5_2018 import AccousticEmissionDataSet_16_5_2018
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import optimizers
from keras.datasets import mnist
# from ideal_dataset import noise_time_shift_dataset

data = np.array([[[1, 2],
                  [3, 4]],
                 [[2, 3],
                  [4, 5]],
                 [[3, 4],
                  [5, 6]],
                 [[11, 12],
                  [13, 14]],
                 [[12, 13],
                  [14, 15]],
                 [[13, 14],
                  [15, 16]]])
print(data)
print(data.shape)

data2 = data

print(data2.shape)

l = []
l.append(data)
# l.append(data2)


z = np.concatenate(l, axis=2)
print(z.shape)
print(z)





