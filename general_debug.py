import numpy as np
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from dataset_experiment_16_5_2018 import AccousticEmissionDataSet_16_5_2018
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import optimizers

model = Sequential()
# Convolutional layer 1 ------------------------------------------
model.add(Conv2D(filters=50, kernel_size=(5, 5), strides=(1, 1),  # kernel covers 1kHz, 25ms
                 activation='relu', input_shape=(500, 40, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(1, 1)))

# Convolutional layer 2 ------------------------------------------
model.add(Conv2D(filters=70, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
# #
# Convolutional layer 3 ------------------------------------------
model.add(Conv2D(filters=108, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
#
# # Convolutional layer 4 ------------------------------------------
# model.add(Conv2D(filters=150, kernel_size=(5, 2), strides=(1, 1),
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(5, 1), strides=(2, 1)))
# #
# # # # Fully connected ----------------------------------------
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(7, activation='softmax'))

print(model.summary())



# l = np.arange(0, 100, 1).reshape((4, 5, 5))
# m = np.arange(100, 200, 1).reshape((4, 5, 5))
# print(l)
# print(l.shape)
# print(m)
# print(m.shape)
#
# x = np.concatenate((l[0], m[0, :, 1:]), axis=1)
# print(x)
# print(x.shape)
#
# a = True
# b = True
#
# if a or b:
#     print('HI')
#     if a:
#         print('A')
#     if b:
#         print('B')


