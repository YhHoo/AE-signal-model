# this section is for function debugging before integrate in full
# put all into function and do not delete old fn but create another
# version in new function

import numpy as np
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from __future__ import print_function
from keras.datasets import mnist
import matplotlib.pyplot as plt
import keras


# My Own CNN Debugging -------------------------------------------
def my_cnn_version_1():
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


# CNN EXAMPLE ----------------------------------------
# source: http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
def cnn_example_1():
    batch_size = 128
    num_classes = 10
    epochs = 100

    # input image dimensions
    img_x, img_y = 28, 28

    # load the MNIST data set, which already splits into train and test sets for us
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # x->(60000, 28, 28), y->(60000,)


    # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
    # because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
    input_shape = (img_x, img_y, 1)

    # convert the data to the right type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices - this is for use in the
    # categorical_crossentropy loss below
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(y_train.shape)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # unspecified stride means it is set to pool_size by default
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    print(model.summary())

    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))

    history = AccuracyHistory()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    plt.plot(range(1, 11), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()