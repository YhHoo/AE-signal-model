# this is for bash to know the path of the src
import sys
sys.path.append('C:/Users/YH/PycharmProjects/AE-signal-model')

from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras import regularizers
# self lib
from src.utils.helpers import direct_to_dir


def lcp_recognition_binary_model():
    visible_in = Input(shape=(6000, 1))
    conv_1 = Conv1D(5, kernel_size=5, activation='relu')(visible_in)
    maxpool_1 = MaxPooling1D(pool_size=3, strides=2)(conv_1)

    dropout_1 = Dropout(0.4)(maxpool_1)

    conv_2 = Conv1D(20, kernel_size=5, activation='relu')(dropout_1)
    maxpool_2 = MaxPooling1D(pool_size=3, strides=2)(conv_2)

    conv_3 = Conv1D(32, kernel_size=5, activation='relu')(maxpool_2)
    maxpool_3 = MaxPooling1D(pool_size=3, strides=2)(conv_3)

    flatten = Flatten()(maxpool_3)

    dropout_2 = Dropout(0.5)(flatten)
    dense_1 = Dense(10, activation='relu')(dropout_2)
    # dense_2 = Dense(20, activation='relu')(dense_1)
    # dense_3 = Dense(80, activation='relu')(dense_2)
    visible_out = Dense(1, activation='sigmoid')(dense_1)

    model = Model(inputs=visible_in, outputs=visible_out)

    print(model.summary())

    return model


def lcp_recognition_binary_model_2():
    '''
    refer Online, VGG concept

    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(6000, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    :return:
    '''
    model = Sequential()

    model.add(Conv1D(16, 3, activation='relu', input_shape=(6000, 1)))
    model.add(Conv1D(16, 3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))
    model.add(Dropout(0.3))

    # model.add(Conv1D(64, 3, activation='relu'))
    # model.add(Conv1D(64, 3, activation='relu'))
    # model.add(MaxPooling1D(3, strides=2))
    # model.add(Dropout(0.3))

    # model.add(Conv1D(128, 3, activation='relu'))
    # model.add(Conv1D(128, 3, activation='relu'))
    # model.add(MaxPooling1D(3, strides=2))
    # model.add(Dropout(0.3))

    # model.add(Conv1D(256, 3, activation='relu'))
    # model.add(Conv1D(256, 3, activation='relu'))

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))

    # model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model


def lcp_recognition_binary_model_3():
    '''
    Dual layer
    '''
    visible_in = Input(shape=(6000, 1))

    # Part a
    conv_a_1 = Conv1D(32, kernel_size=5, activation='relu', name='conv_a_1')(visible_in)
    conv_a_2 = Conv1D(32, kernel_size=5, activation='relu', name='conv_a_2')(conv_a_1)
    maxpool_a_1 = MaxPooling1D(pool_size=3, strides=2, name='maxp_a_1')(conv_a_2)
    drop_a_1 = Dropout(0.3, name='drop_a_1')(maxpool_a_1)

    conv_a_3 = Conv1D(64, kernel_size=5, activation='relu', name='conv_a_3')(drop_a_1)
    conv_a_4 = Conv1D(128, kernel_size=5, activation='relu', name='conv_a_4', use_bias=False)(conv_a_3)
    maxpool_a_2 = MaxPooling1D(pool_size=3, strides=2, name='maxp_a_2')(conv_a_4)

    gap_a_1 = GlobalAveragePooling1D(name='gap_a_1')(maxpool_a_2)

    # Part b
    conv_b_1 = Conv1D(32, kernel_size=5, activation='relu', name='conv_b_1')(visible_in)
    conv_b_2 = Conv1D(32, kernel_size=5, activation='relu', name='conv_b_2')(conv_b_1)
    maxpool_b_1 = MaxPooling1D(pool_size=3, strides=2, name='maxp_b_1')(conv_b_2)
    drop_b_1 = Dropout(0.3, name='drop_b_1')(maxpool_b_1)
    conv_b_3 = Conv1D(128, kernel_size=5, activation='relu', name='conv_b_3')(drop_b_1)
    # drop_b_2 = Dropout(0.3, name='drop_b_2')(conv_b_3)
    # conv_b_4 = Conv1D(128, kernel_size=5, activation='relu', name='conv_b_4')(drop_b_2)
    # maxpool_b_2 = MaxPooling1D(pool_size=3, strides=2, name='maxp_b_2')(conv_b_4)

    gap_b_1 = GlobalAveragePooling1D(name='gap_b_1')(conv_b_3)

    # Layer 2
    merge_1 = concatenate([gap_a_1, gap_b_1])
    dense_1 = Dense(50, activation='relu', name='dense_1')(merge_1)
    drop_1 = Dropout(0.2, name='drop_1')(dense_1)
    visible_out = Dense(1, activation='sigmoid', name='dense_2')(drop_1)

    model = Model(inputs=visible_in, outputs=visible_out)

    print(model.summary())

    save_model_plot = direct_to_dir(where='result') + 'lcp_recognition_binary_model_3.png'
    plot_model(model, to_file=save_model_plot)

    return model


def lcp_recognition_binary_model_4():
    model = Sequential()
    model.add(Dense(108, activation='relu', input_dim=6000))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    print(model.summary())

    return model


# LCP RECOGNITION BY DISTANCE ------------------------------------------------------------------------------------------
def lcp_by_dist_recognition_multi_model_1():
    model = Sequential()

    model.add(Conv1D(16, 3, activation='relu', input_shape=(6000, 1)))
    model.add(Conv1D(16, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(3, strides=2))

    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(3, strides=2))

    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(3, strides=2))

    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(3, strides=2))

    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))

    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    print(model.summary())

    return model


def lcp_by_dist_recognition_multi_model_2():
    model = Sequential()

    model.add(Conv1D(16, 3, activation='relu', input_shape=(6000, 1)))
    model.add(Conv1D(16, 3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))

    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))

    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))

    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))

    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))  # try tanh
    model.add(Dense(2, activation='softmax'))

    print(model.summary())

    return model


# LEAK NO LEAK BY DISTANCE------------------------------------------------------------------------------------------

def LNL_binary_model():
    model = Sequential()

    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(6000, 1)))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))

    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))

    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))

    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))  # try tanh
    model.add(Dense(2, activation='softmax'))

    print(model.summary())

    return model


def dexter_model():
    inp = Input((6000, 1))
    # 256
    x = BatchNormalization()(inp)

    # 256
    x = Conv1D(32, kernel_size=3, dilation_rate=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(32, kernel_size=3, dilation_rate=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    #  128
    x = Conv1D(64, kernel_size=3, dilation_rate=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x_prev = x
    x = Conv1D(64, kernel_size=3, dilation_rate=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Add()([x, x_prev])
    x_prev = x

    x = Conv1D(64, kernel_size=3, dilation_rate=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, x_prev])
    x_prev = x

    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # 64
    x = Conv1D(128, kernel_size=3, dilation_rate=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x_prev = x
    x = Conv1D(128, kernel_size=3, dilation_rate=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, x_prev])
    x_prev = x

    x = Conv1D(128, kernel_size=3, dilation_rate=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, x_prev])
    x_prev = x

    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    #  32
    x = Conv1D(128, kernel_size=3, dilation_rate=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x_prev = x
    x = Conv1D(128, kernel_size=3, dilation_rate=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, x_prev])
    x_prev = x

    x = Conv1D(128, kernel_size=3, dilation_rate=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, x_prev])
    x_prev = x

    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    #  16
    x = Conv1D(256, kernel_size=3, dilation_rate=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x_prev = x
    x = Conv1D(256, kernel_size=3, dilation_rate=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, x_prev])
    x_prev = x

    x = Conv1D(256, kernel_size=3, dilation_rate=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, x_prev])

    x = GlobalMaxPooling1D()(x)

    x = Dropout(0.55)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)

    out = Dense(2, activation='softmax')(x)

    model = Model(inp, out)
    print(model.summary())

    return model


def LNL_binary_model_2():
    inp = Input((2000, 1))

    x = BatchNormalization()(inp)

    # conv 1
    x = Conv1D(filters=32, kernel_size=200, strides=1, dilation_rate=1, activation='relu', padding='same')(x)  # kernel size of 0.0005s
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)  # time half

    # # conv 2
    x = Conv1D(filters=64, kernel_size=200, strides=1, dilation_rate=1, activation='relu', padding='same')(x)  # kernel size of 0.0001s
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)  # time half

    # conv 3
    x = Conv1D(filters=128, kernel_size=100, strides=1, dilation_rate=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # # conv 4
    # x = Conv1D(filters=256, kernel_size=100, strides=1, activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.40)(x)

    x = Dense(240, activation='relu')(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    out = Dense(2, activation='softmax')(x)

    model = Model(inp, out)

    print(model.summary())

    return model


model = LNL_binary_model_2()


# # --------------------HERE FOR TESTING THE MODEL ALLOWABLE BATCH SIZE FOR GPU MEMORY LIMIT -----------------------------
# from src.utils.helpers import *
# data = np.random.rand(100000, 2000)
# data = data.reshape((data.shape[0], data.shape[1], 1))
# label = np.ones(100000).reshape((100000, -1))
# label = to_categorical(label, num_classes=2)
#
#
# model = LNL_binary_model_2()
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# model.fit(x=data,
#           y=label,
#           validation_split=0.7,
#           epochs=100,
#           shuffle=True,
#           verbose=2,
#           batch_size=1000)

