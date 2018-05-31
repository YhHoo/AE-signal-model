import numpy as np
from keras.utils import to_categorical
# self declared library
from ideal_dataset import noise_time_shift_dataset
from utils import break_into_train_test, ModelLogger
from cnn_model_bank import cnn_2_51_3class_v1

# time axis setting
fs = 1000
duration = 10  # tune this for duration
total_point = int(fs * duration)
time_axis = np.linspace(0, duration, total_point)
dataset, label = noise_time_shift_dataset(time_axis, fs=fs, verbose=True)

train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=3,
                                                         train_split=0.7,
                                                         verbose=True)

# reshape to satisfy conv2d input shape
train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))

# convert
train_y = to_categorical(train_y, num_classes=3)
test_y = to_categorical(test_y, num_classes=3)


model = cnn_2_51_3class_v1()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_logger = ModelLogger(model, model_name='white_noise_cnn')

history = model.fit(x=train_x,
                    y=train_y,
                    batch_size=10,
                    epochs=50,
                    verbose=2,
                    validation_data=(test_x, test_y))

model_logger.learning_curve(history=history, show=True)
