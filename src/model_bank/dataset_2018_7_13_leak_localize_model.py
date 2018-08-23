'''
ABSTRACT:
Models here aim to classify the leak pattern (in 2d xcor map, 1d xcor map, xcor_max_vec, etc.) into classes of
distance difference between leak source to 2 AE sensors
'''
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, AveragePooling2D
from keras.models import Sequential, Model


# compact shallow NN
def fc_leak_1bar_max_vec_v1(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(800, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())

    return model


# more deeper NN
def fc_leak_1bar_max_vec_v2(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # print(model.summary())

    return model


# fc_leak_1bar_max_vec(input_shape=50, num_classes=11)
