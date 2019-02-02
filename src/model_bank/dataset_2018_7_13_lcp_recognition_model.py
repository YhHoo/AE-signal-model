# this is for bash to know the path of the src
import sys
sys.path.append('C:/Users/YH/PycharmProjects/AE-signal-model')

from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras import regularizers
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
    '''
    kernel & bias l2 reg, 3 conv layer (BEST MODEL SO FAR, UNDER LNL_21x3 & LNL_29x1)
    BEST MODEL FOR ds dataset
    '''
    print('MODEL: LNL_29x1 Best MODEL')
    inp = Input((2000, 1))

    x = BatchNormalization()(inp)

    # conv 1
    x = Conv1D(filters=32, kernel_size=200, strides=1, dilation_rate=1, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # # conv 2
    x = Conv1D(filters=64, kernel_size=200, strides=1, dilation_rate=1, padding='same',
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # conv 3
    x = Conv1D(filters=128, kernel_size=100, strides=1, dilation_rate=1, padding='same',
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.55)(x)

    x = Dense(240, activation='relu', )(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    out = Dense(2, activation='softmax')(x)

    model = Model(inp, out)

    print(model.summary())

    return model


def LNL_binary_model_3():
    '''
    Duplicate of LNL_29x1, except a smaller kernel size, for ds2
    '''
    inp = Input((2000, 1))

    x = BatchNormalization()(inp)

    # conv 1
    x = Conv1D(filters=32, kernel_size=20, strides=1, dilation_rate=1, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # # conv 2
    x = Conv1D(filters=64, kernel_size=20, strides=1, dilation_rate=1, padding='same',
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # conv 3
    x = Conv1D(filters=128, kernel_size=10, strides=1, dilation_rate=1, padding='same',
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.55)(x)

    x = Dense(240, activation='relu', )(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    out = Dense(2, activation='softmax')(x)

    model = Model(inp, out)

    print(model.summary())

    return model


def LNL_binary_model_4():
    '''
    dual activation, 4 conv layer with l2 reg, kernel for ds2
    '''
    inp = Input((2000, 1))

    x = BatchNormalization()(inp)

    # conv 1
    x = Conv1D(filters=32, kernel_size=20, strides=1, dilation_rate=1, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # # conv 2
    x = Conv1D(filters=64, kernel_size=20, strides=1, dilation_rate=1, padding='same',
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # conv 3
    x = Conv1D(filters=128, kernel_size=10, strides=1, dilation_rate=1, padding='same',
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = Conv1D(filters=128, kernel_size=10, strides=1, dilation_rate=1, padding='same',
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.55)(x)

    x = Dense(240, activation='relu', )(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    out = Dense(2, activation='softmax')(x)

    model = Model(inp, out)

    print(model.summary())

    return model


def LNL_binary_model_5():
    '''
    A multipath
    :return:
    '''
    inp_1 = Input((2000, 1))
    inp_2 = Input((2000, 1))

    x_1 = BatchNormalization()(inp_1)
    x_2 = BatchNormalization()(inp_2)

    # path 1 -----
    x_1 = Conv1D(filters=32, kernel_size=200, strides=1, dilation_rate=1, padding='same')(x_1)
    x_1 = Activation('relu')(x_1)
    x_1 = BatchNormalization()(x_1)
    x_1 = MaxPooling1D(pool_size=3, strides=2, padding='same')(x_1)

    x_1 = Conv1D(filters=64, kernel_size=200, strides=1, dilation_rate=1, padding='same')(x_1)
    x_1 = Activation('relu')(x_1)
    x_1 = BatchNormalization()(x_1)
    x_1 = MaxPooling1D(pool_size=2, strides=2, padding='same')(x_1)

    x_1 = Conv1D(filters=128, kernel_size=100, strides=1, dilation_rate=1, padding='same')(x_1)
    x_1 = Activation('relu')(x_1)
    x_1 = BatchNormalization()(x_1)
    x_1 = MaxPooling1D(pool_size=2, strides=2, padding='same')(x_1)

    # path 2 -----
    x_2 = Conv1D(filters=32, kernel_size=200, strides=1, dilation_rate=1, padding='same')(x_2)
    x_2 = Activation('relu')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = MaxPooling1D(pool_size=3, strides=2, padding='same')(x_2)

    x_2 = Conv1D(filters=64, kernel_size=200, strides=1, dilation_rate=1, padding='same')(x_2)
    x_2 = Activation('relu')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = MaxPooling1D(pool_size=2, strides=2, padding='same')(x_2)

    x_2 = Conv1D(filters=128, kernel_size=100, strides=1, dilation_rate=1, padding='same')(x_2)
    x_2 = Activation('relu')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = MaxPooling1D(pool_size=2, strides=2, padding='same')(x_2)

    # concate
    x_c = concatenate([x_1, x_2])
    x_c = BatchNormalization()(x_c)

    x_c = Conv1D(filters=32, kernel_size=200, strides=1, dilation_rate=1, padding='same')(x_c)
    x_c = Activation('relu')(x_c)
    x_c = BatchNormalization()(x_c)
    x_c = MaxPooling1D(pool_size=3, strides=2, padding='same')(x_c)

    x_c = Conv1D(filters=64, kernel_size=100, strides=1, dilation_rate=1, padding='same')(x_c)
    x_c = Activation('relu')(x_c)
    x_c = BatchNormalization()(x_c)
    x_c = MaxPooling1D(pool_size=3, strides=2, padding='same')(x_c)

    x_c = GlobalAveragePooling1D()(x_c)
    x_c = Dropout(0.55)(x_c)

    x_c = Dense(240, activation='relu')(x_c)
    x_c = Dense(120, activation='relu')(x_c)
    x_c = Dense(2, activation='softmax')(x_c)

    out = Dense(2, activation='softmax')(x_c)

    model = Model([inp_1, inp_2], out)

    print(model.summary())

    return model


def LNL_binary_model_6():
    '''
    RESNET shortcut connection and using preactivation
    '''
    inp = Input((2000, 1))

    x = BatchNormalization()(inp)

    # conv block 1
    x = Activation('relu')(x)
    x = Conv1D(filters=32, kernel_size=200, strides=1, dilation_rate=1, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_store = x
    x = Conv1D(filters=32, kernel_size=200, strides=1, dilation_rate=1, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Add()([x, x_store])
    x_store = x
    x = Conv1D(filters=32, kernel_size=200, strides=1, dilation_rate=1, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Add()([x, x_store])

    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)  # -------

    # # conv 2
    x = Conv1D(filters=64, kernel_size=200, strides=1, dilation_rate=1, padding='same',
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # conv 3
    x = Conv1D(filters=128, kernel_size=100, strides=1, dilation_rate=1, padding='same',
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # # conv block 2
    # x = Activation('relu')(x)
    # x = Conv1D(filters=64, kernel_size=200, strides=1, dilation_rate=1, padding='same',
    #            kernel_regularizer=regularizers.l2(0.01))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x_store = x
    # x = Conv1D(filters=64, kernel_size=200, strides=1, dilation_rate=1, padding='same',
    #            kernel_regularizer=regularizers.l2(0.01))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Add()([x, x_store])
    # x_store = x
    # x = Conv1D(filters=64, kernel_size=200, strides=1, dilation_rate=1, padding='same',
    #            kernel_regularizer=regularizers.l2(0.01))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Add()([x, x_store])

    # x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)  # -------
    #
    # # conv block 3
    # x = Activation('relu')(x)
    # x = Conv1D(filters=128, kernel_size=200, strides=1, dilation_rate=1, padding='same',
    #            kernel_regularizer=regularizers.l2(0.01))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x_store = x
    # x = Conv1D(filters=128, kernel_size=200, strides=1, dilation_rate=1, padding='same',
    #            kernel_regularizer=regularizers.l2(0.01))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Add()([x, x_store])
    # x_store = x
    # x = Conv1D(filters=128, kernel_size=200, strides=1, dilation_rate=1, padding='same',
    #            kernel_regularizer=regularizers.l2(0.01))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Add()([x, x_store])

    # x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.55)(x)

    x = Dense(240, activation='relu', )(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    out = Dense(2, activation='softmax')(x)

    model = Model(inp, out)

    print(model.summary())

    return model


# LNL_binary_model_5()
# --------------------HERE FOR TESTING THE MODEL ALLOWABLE BATCH SIZE FOR GPU MEMORY LIMIT -----------------------------

# from src.utils.helpers import *
# data_1 = np.random.rand(10000, 2000)
# data_1 = data_1.reshape((data_1.shape[0], data_1.shape[1], 1))
#
# data_2 = np.random.rand(10000, 2000)
# data_2 = data_1.reshape((data_2.shape[0], data_2.shape[1], 1))
#
# label_1 = np.ones(10000).reshape((10000, -1))
# label_1 = to_categorical(label_1, num_classes=2)
#
# model = LNL_binary_model_3()
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# model.fit(x=[data_1, data_2],
#           y=label_1,
#           validation_split=0.7,
#           epochs=100,
#           shuffle=True,
#           verbose=2,
#           batch_size=100)

