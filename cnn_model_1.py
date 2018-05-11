import numpy as np
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

x_train = np.random.rand(3000, 100)

# reshape
x_train = x_train.reshape((1, 3000, 100, 1))
print(x_train.shape)

y_train = 1
y_train = to_categorical(y_train, num_classes=12).reshape((1, -1))
print(y_train.shape)

model = Sequential()
# Convolutional layer 1 ------------------------------------------
model.add(Conv2D(filters=36, kernel_size=(10, 10), strides=(1, 1),
                 activation='relu', input_shape=(3000, 100, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
# Convolutional layer 2 ------------------------------------------
model.add(Conv2D(filters=72, kernel_size=(10, 5), strides=(2, 1),
                 activation='relu', input_shape=(3000, 100, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
# Convolutional layer 3 ------------------------------------------
model.add(Conv2D(filters=96, kernel_size=(10, 8), strides=(4, 1),
                 activation='relu', input_shape=(3000, 100, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(3, 3)))
# Convolutional layer 4 ------------------------------------------
model.add(Conv2D(filters=109, kernel_size=(6, 2), strides=(1, 1),
                 activation='relu', input_shape=(3000, 100, 1)))
model.add(MaxPooling2D(pool_size=(9, 2), strides=(2, 1)))
model.add(Flatten())
# Fully connected layer 1 ----------------------------------------
model.add(Dense(150, activation='relu'))
# Fully connected layer 1 ----------------------------------------
model.add(Dense(80, activation='relu'))
model.add(Dense(2, activation='softmax'))
# print architecture summary
print(model.summary())

# training
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=10, verbose=2)