'''
Model Naming Practice: [model architecture]_[input dimension]_[class]_[variant]_[comment(optional)]
e.g. cnn_3000_40_2class_v1 or cnn_3000_40__7class_v1_dropout
'''

from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, AveragePooling2D
from keras.models import Sequential, Model
# self defined library


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
    Input: the xcor map produced from 2 phase maps of white noise of diff lags.
    Result: Able to recognize a 51x169 xcor map perfectly.
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
    model.add(Dense(200, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    print(model.summary())

    return model


def cnn2d_plb_v1(input_shape, num_classes):
    '''
    This CNN takes in 2d segmented xcor map of PLB signals.
    Classify to their distance difference btw leak and 2 sensors
    Used on:
    AcousticEmissionDataSet_13_7_2018.plb()
    '''
    model = Sequential()

    model.add(Conv2D(filters=5, kernel_size=(2, 10), strides=(1, 1),
                     activation='relu', input_shape=(input_shape[0], input_shape[1], 1)))

    model.add(Conv2D(filters=8, kernel_size=(2, 10), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 5), strides=(2, 2)))

    model.add(Conv2D(filters=8, kernel_size=(2, 10), strides=(1, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 5), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.2))

    # Fully connected ----------------------------------------
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())

    return model


def cnn1d_plb_v1(input_shape, num_classes):
    inputs = Input(shape=(784,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)


# cnn2d_plb_v1(input_shape=(10, 300), num_classes=41)



# inputs = Input(shape=(784,))
#
# # a layer instance is callable on a tensor, and returns a tensor
# x = Dense(64, activation='relu')(inputs)
# x = Dense(64, activation='relu')(x)
# predictions = Dense(10, activation='softmax')(x)
#
# # This creates a model that includes
# # the Input layer and three Dense layers
# model = Model(inputs=inputs, outputs=predictions)
#
# print(model.summary())