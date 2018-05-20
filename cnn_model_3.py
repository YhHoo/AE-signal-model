# ------------------------------------------------------
# this model expecting the input data of 0.2 seconds with shape of (3000, 40) only.
# ------------------------------------------------------

from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from dataset_experiment_16_5_2018 import AccousticEmissionDataSet_16_5_2018
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import optimizers

# data set
ae_data = AccousticEmissionDataSet_16_5_2018()
train_x, train_y, test_x, test_y = ae_data.sleak_1bar_7pos()

# reshape to satisfy conv2d input shape
train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))

# convert
train_y = to_categorical(train_y, num_classes=7)
test_y = to_categorical(test_y, num_classes=7)


# data summary
print('\n----------INPUT DATA DIMENSION---------')
print('Train_X dim: ', train_x.shape)
print('Train_Y dim: ', train_y.shape)
print('Test_X dim: ', test_x.shape)
print('Test_Y dim: ', test_y.shape)


# time-res = 5ms per band, f-res = 100Hz per band
model = Sequential()
# Convolutional layer 1 ------------------------------------------
model.add(Conv2D(filters=36, kernel_size=(10, 5), strides=(1, 1),  # kernel covers 1kHz, 25ms
                 activation='relu', input_shape=(3000, 40, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(1, 1)))

# Convolutional layer 2 ------------------------------------------
model.add(Conv2D(filters=72, kernel_size=(10, 5), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

# Convolutional layer 3 ------------------------------------------
model.add(Conv2D(filters=108, kernel_size=(10, 5), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(3, 2)))

# Convolutional layer 4 ------------------------------------------
model.add(Conv2D(filters=150, kernel_size=(10, 2), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 1), strides=(4, 1)))

# Fully connected ----------------------------------------
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(7, activation='softmax'))

print(model.summary())

adam_optimizer = optimizers.Adam(lr=0.01)
model.compile(optimizer='adam', loss='categorical_crossentropy')

history = model.fit(x=train_x,
                    y=train_y,
                    validation_data=(test_x, test_y),
                    batch_size=30,
                    epochs=10,
                    shuffle=True)

# visualize of training process
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.legend()
plt.show()
