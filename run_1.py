import numpy as np
from keras.utils import to_categorical
# self declared library
from ideal_dataset import noise_time_shift_dataset
from utils import break_into_train_test, ModelLogger, model_multiclass_evaluate, reshape_3d_to_4d_tocategorical
from cnn_model_bank import cnn_2_51_3class_v1

# ------------------------------------------------------------------------------------- RUN_1
'''
Data set: Randomly Generated White Noise with 3 shifts i.e. 0.1s, 0.2s and 0.3s. 
Data Pre-processing: Phase info from STFT, 3 classes, CNN input shape of (n, 2, 51, 1)
'''
# time axis setting
fs = 1000
duration = 10  # tune this for duration
total_point = int(fs * duration)
time_axis = np.linspace(0, duration, total_point)

dataset, label = noise_time_shift_dataset(time_axis, fs=fs, verbose=True)


train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=3,
                                                         train_split=0.8,
                                                         verbose=True)

# reshape to satisfy conv2d input shape
train_x, train_y, test_x, test_y = reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y,
                                                                  fourth_dim=1, num_classes=3, verbose=True)

# model building
model = cnn_2_51_3class_v1()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_logger = ModelLogger(model, model_name='white_noise_cnn')

# model training
history = model.fit(x=train_x,
                    y=train_y,
                    batch_size=30,
                    epochs=300,
                    verbose=2,
                    validation_data=(test_x, test_y))

model_logger.learning_curve(history=history, show=True)
model_multiclass_evaluate(model, test_x=test_x, test_y=test_y)

