from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, Dropout, GlobalAveragePooling1D, AveragePooling2D, Conv1D
from keras.models import Sequential, Model


def lcp_recognition_binary_model():
    visible_in = Input(shape=(6000, 1))
    conv_1 = Conv1D(5, kernel_size=5, activation='relu')(visible_in)
    maxpool_1 = MaxPooling1D(pool_size=2, strides=2)(conv_1)

    # conv_2 = Conv1D(32, kernel_size=5, activation='relu')(maxpool_1)
    # maxpool_2 = MaxPooling1D(pool_size=2, strides=2)(conv_2)
    #
    # conv_3 = Conv1D(32, kernel_size=5, activation='relu')(maxpool_2)
    # maxpool_3 = MaxPooling1D(pool_size=2, strides=2)(conv_3)

    flatten = Flatten()(maxpool_1)

    dense_1 = Dense(10, activation='relu')(flatten)
    visible_out = Dense(1, activation='sigmoid')(dense_1)

    model = Model(inputs=visible_in, outputs=visible_out)

    print(model.summary())

    return model


def lcp_recognition_binary_model_2():
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(6000, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model