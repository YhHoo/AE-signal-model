# ------------------------------------------------------
# this model expecting the input data of 0.2 seconds with shape of (3000/1000/700, 40) only.
# ------------------------------------------------------

from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from dataset_experiment_16_5_2018 import AccousticEmissionDataSet_16_5_2018
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint

# ----------------------------------------------------------------------------------------------TEST 1

# data set
ae_data = AccousticEmissionDataSet_16_5_2018()
train_x, train_y, test_x, test_y = ae_data.sleak_1bar_7pos(f_range=(0, 1000))

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

# # Fully connected ----------------------------------------
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(7, activation='softmax'))

print(model.summary())

adam_optimizer = optimizers.Adam(lr=0.01)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# checkpoint- this will save the model during training every time the accuracy hits a new highest
filepath = 'CNN_Test_1.h5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',  # for acc, it should b 'max'; for loss, 'min'
                             period=1)  # no of epoch btw checkpoints
callback_list = [checkpoint]

history = model.fit(x=train_x,
                    y=train_y,
                    validation_data=(test_x, test_y),
                    batch_size=30,
                    epochs=10,
                    shuffle=True,
                    callbacks=callback_list)

# visualize of training process
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.legend()
plt.title('F-range: 0-100kHz')
plt.savefig('result/test_1.png')
plt.close()

# ----[Saving Model]----
# serialize and saving the model structure to JSON
model_name = 'CNN_Test_1'
model_json = model.to_json()
with open(model_name + '.json', 'w') as json_file:
    json_file.write(model_json)

# ----------------------------------------------------------------------------------------------TEST 2

# data set
ae_data = AccousticEmissionDataSet_16_5_2018()
train_x, train_y, test_x, test_y = ae_data.sleak_1bar_7pos(f_range=(0, 700))

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

model = Sequential()
# Convolutional layer 1 ------------------------------------------
model.add(Conv2D(filters=50, kernel_size=(5, 5), strides=(1, 1),  # kernel covers 1kHz, 25ms
                 activation='relu', input_shape=(700, 40, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(1, 1)))

# Convolutional layer 2 ------------------------------------------
model.add(Conv2D(filters=70, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
#
# Convolutional layer 3 ------------------------------------------
model.add(Conv2D(filters=108, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

# Convolutional layer 4 ------------------------------------------
model.add(Conv2D(filters=150, kernel_size=(5, 2), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 1), strides=(2, 1)))
#
# # # Fully connected ----------------------------------------
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(7, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy')

filepath = 'CNN_Test_2.h5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',  # for acc, it should b 'max'; for loss, 'min'
                             period=1)  # no of epoch btw checkpoints

history = model.fit(x=train_x,
                    y=train_y,
                    validation_data=(test_x, test_y),
                    batch_size=30,
                    epochs=10,
                    shuffle=True,
                    callbacks=callback_list)

# visualize of training process
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.legend()
plt.title('F-range: 0-70kHz')
plt.savefig('result/test_2.png')
plt.close()

# ----[Saving Model]----
# serialize and saving the model structure to JSON
model_name = 'CNN_Test_2'
model_json = model.to_json()
with open(model_name + '.json', 'w') as json_file:
    json_file.write(model_json)