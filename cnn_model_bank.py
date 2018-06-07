from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout, LocallyConnected2D
from keras.models import Sequential
from dataset_experiment_16_5_2018 import AccousticEmissionDataSet_16_5_2018
from keras.utils import to_categorical
from keras import optimizers
# self defined library
from utils import ModelLogger, model_multiclass_evaluate

'''
Model Naming Practice: [model architecture]_[input dimension]_[class]_[variant]_[comment(optional)]
e.g. cnn_3000_40_2class_v1 or cnn_3000_40__7class_v1_dropout
'''


def cnn_1000_40_7class_v1():
    '''
    :return: a CNN that has input shape of (1000, 40) with 1.6M trainable param
    '''
    model = Sequential()

    # Convolutional layer 1 ------------------------------------------
    model.add(Conv2D(filters=40, kernel_size=(5, 5), strides=(1, 1),  # kernel covers 1kHz, 25ms
                     activation='relu', input_shape=(1000, 40, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(1, 1)))

    # Convolutional layer 2 ------------------------------------------
    model.add(Conv2D(filters=60, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
    #
    # Convolutional layer 3 ------------------------------------------
    model.add(Conv2D(filters=108, kernel_size=(5, 5), strides=(2, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # Convolutional layer 4 ------------------------------------------
    model.add(Conv2D(filters=150, kernel_size=(5, 2), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 1), strides=(2, 1)))

    # Fully connected ----------------------------------------
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    return model


def cnn_700_40_7class_v1():
    '''
    :return: a CNN that has input shape of (700, 40) with 1.3M trainable param
    '''
    model = Sequential()

    # Convolutional layer 1 ------------------------------------------
    model.add(Conv2D(filters=50, kernel_size=(5, 5), strides=(1, 1),  # kernel covers 1kHz, 25ms
                     activation='relu', input_shape=(700, 40, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(1, 1)))

    # Convolutional layer 2 ------------------------------------------
    model.add(Conv2D(filters=70, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # Convolutional layer 3 ------------------------------------------
    model.add(Conv2D(filters=108, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # Convolutional layer 4 ------------------------------------------
    model.add(Conv2D(filters=150, kernel_size=(5, 2), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 1), strides=(2, 1)))

    # Fully connected ----------------------------------------
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    return model


def cnn_3000_100_12class_v1():
    '''
    :return: a CNN that has input shape of (3000, 100) with 0.95M trainable param
    '''
    model = Sequential()

    # Convolutional layer 1 ------------------------------------------
    model.add(Conv2D(filters=36, kernel_size=(10, 10), strides=(1, 1),
                     activation='relu', input_shape=(3000, 100, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # Convolutional layer 2 ------------------------------------------
    model.add(Conv2D(filters=72, kernel_size=(10, 5), strides=(2, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # Convolutional layer 3 ------------------------------------------
    model.add(Conv2D(filters=96, kernel_size=(10, 8), strides=(4, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(3, 3)))

    # Convolutional layer 4 ------------------------------------------
    model.add(Conv2D(filters=109, kernel_size=(6, 2), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(9, 2), strides=(2, 1)))
    model.add(Flatten())

    # Fully connected layer 1 ----------------------------------------
    model.add(Dense(150, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(12, activation='softmax'))


def cnn_3000_23_2class_v1():
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

    # Fully connected ----------------------------------------
    model.add(Dense(150, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(2, activation='softmax'))


def cnn_2_51_3class_v1(fc):
    '''
    Input: phase map of 2 sensors concatenate side by side,
    where 2 means sensor no, 51 is frequency bin
    '''
    model = Sequential()

    # Convolutional layer 1 ------------------------------------------
    model.add(Conv2D(filters=60, kernel_size=(2, 4), strides=(1, 1),
                     activation='relu', input_shape=(2, 51, 1)))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 1)))
    model.add(Dropout(0.2))

    # Convolutional layer 2 ------------------------------------------
    model.add(Conv2D(filters=100, kernel_size=(1, 2), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 1)))
    model.add(Dropout(0.4))

    # Convolutional layer 3 ------------------------------------------
    # model.add(Conv2D(filters=96, kernel_size=(1, 3), strides=(4, 1),
    #                  activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 1)))
    # model.add(Dropout(0.3))

    # Convolutional layer 4 ------------------------------------------
    # model.add(Conv2D(filters=109, kernel_size=(1, 3), strides=(1, 1),
    #                  activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 1)))

    # Flatten all into 1d vector--------------------------------------
    model.add(Flatten())
    model.add(Dropout(0.4))

    # Fully connected ----------------------------------------
    model.add(Dense(fc[0], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(fc[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(fc[2], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    print(model.summary())

    return model


def fc_2x51_3class(fc):
    '''
    Input: Flatten phase map of 2 sensors concatenate side by side,
    where 2 means sensor no, 51 is frequency bin.
    '''
    model = Sequential()

    # Fully connected ----------------------------------------
    model.add(Dense(fc[0], activation='relu', input_dim=102))
    model.add(Dropout(0.3))
    model.add(Dense(fc[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(fc[2], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(fc[3]))
    model.add(Dropout(0.2))
    model.add(Dense(fc[4]))
    # output layer
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    print(model.summary())

    return model


# this model gives test_acc of 0.98 in first epoch
def cnn_28_28_mnist_10class():
    model = Sequential()

    model.add(Conv2D(filters=30, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=10, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    print(model.summary())

    return model


def cnn_51_159_3class_v1():
    '''
    Input: phase map of 2 sensors concatenate side by side,
    where 2 means sensor no, 51 is frequency bin
    '''
    model = Sequential()

    # Convolutional layer 1 ------------------------------------------
    model.add(Conv2D(filters=36, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu', input_shape=(51, 159, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    # model.add(Dropout(0.2))

    # Convolutional layer 2 ------------------------------------------
    model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.4))

    # Convolutional layer 3 ------------------------------------------
    model.add(Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.3))

    # Convolutional layer 4 ------------------------------------------
    # model.add(Conv2D(filters=109, kernel_size=(1, 3), strides=(1, 1),
    #                  activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 1)))

    # Flatten all into 1d vector--------------------------------------
    model.add(Flatten())
    model.add(Dropout(0.4))

    # Fully connected ----------------------------------------
    model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(fc[1], activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(fc[2], activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(3, activation='softmax'))

    print(model.summary())

    return model


cnn_51_159_3class_v1()