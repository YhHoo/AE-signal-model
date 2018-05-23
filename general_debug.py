import numpy as np
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from dataset_experiment_16_5_2018 import AccousticEmissionDataSet_16_5_2018
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import optimizers
from keras.datasets import mnist


# l = [[1, 0, 0, 0],
#      [0, 1, 0, 0],
#      [0, 0, 1, 0],
#      [0, 0, 0, 1],
#      [0, 1, 0, 0]]
# x = np.argmax(l, axis=1)
# print(x.shape)
# print(x)

# plt.plot(x, marker='x')
# plt.show()

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


