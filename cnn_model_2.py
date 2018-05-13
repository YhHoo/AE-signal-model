# this model expecting the input data of 0.2 seconds with shape of (3000, 23) only.

import numpy as np
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical


model = Sequential()
# Convolutional layer 1 ------------------------------------------
model.add(Conv2D(filters=36, kernel_size=(10, 5), strides=(1, 1),
                 activation='relu', input_shape=(3000, 23, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

# Convolutional layer 2 ------------------------------------------
model.add(Conv2D(filters=72, kernel_size=(10, 5), strides=(2, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 2), strides=(2, 1)))

# Convolutional layer 3 ------------------------------------------
model.add(Conv2D(filters=96, kernel_size=(10, 3), strides=(4, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 1), strides=(3, 1)))

# Convolutional layer 4 ------------------------------------------
model.add(Conv2D(filters=109, kernel_size=(6, 1), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(9, 1), strides=(2, 1)))
model.add(Flatten())

# Fully connected layer 1 ----------------------------------------
model.add(Dense(150, activation='relu'))

# Fully connected layer 2 ----------------------------------------
model.add(Dense(80, activation='relu'))

# Fully connected layer 3 ----------------------------------------
model.add(Dense(2, activation='softmax'))
# print architecture summary
print(model.summary())

