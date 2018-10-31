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
    # model.add(Conv1D(16, 3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(32, 3, activation='relu'))
    # model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(64, 3, activation='relu'))
    # model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3, strides=2))
    model.add(Dropout(0.3))

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






