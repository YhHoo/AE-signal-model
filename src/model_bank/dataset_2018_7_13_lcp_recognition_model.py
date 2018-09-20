from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, Dropout, GlobalAveragePooling2D, AveragePooling2D, Conv1D
from keras.models import Sequential, Model


input = Input(shape=(6000, 1, ))
conv_1 = Conv1D(32, kernel_size=10, activation='relu')(input)
maxpool_1 = MaxPooling1D(pool_size=2, strides=2)(conv_1)
conv_2 = Conv1D(64, kernel_size=10, activation='relu')(maxpool_1)
maxpool_2 = MaxPooling1D(pool_size=2, strides=2)(conv_2)
flatten = Flatten()(maxpool_2)
dense_1 = Dense(50, activation='relu')(flatten)
output = Dense(1, activation='softmax')(dense_1)

model = Model(inputs=input, outputs=output)

print(model.summary())
