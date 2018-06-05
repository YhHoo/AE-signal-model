'''
Data set: Randomly Generated White Noise with 3 shifts i.e. 0.1s, 0.2s and 0.3s.
Data Pre-processing: Phase info from STFT, 3 classes, CNN input shape of (n, 2, 51, 1)
'''

import numpy as np
# self declared library
from ideal_dataset import noise_time_shift_dataset
from utils import break_into_train_test, ModelLogger, model_multiclass_evaluate, reshape_3d_to_4d_tocategorical
from cnn_model_bank import cnn_2_51_3class_v1

# ------------------------------------------------------------------------------------- Dataset 1


# time axis setting
fs = 1000
duration = 20  # tune this for duration
total_point = int(fs * duration)
time_axis = np.linspace(0, duration, total_point)

dataset, label = noise_time_shift_dataset(time_axis, fs=fs, verbose=True, num_series=1)

train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=3,
                                                         train_split=0.7,
                                                         verbose=True)

# reshape to satisfy conv2d input shape
train_x, train_y, test_x, test_y = reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y,
                                                                  fourth_dim=1, num_classes=3, verbose=True)
# Testing different FC architecture
fc_list = [[200, 100, 50], [250, 150, 100], [350, 200, 150], [400, 300, 200]]
model_name = ['FC_200_100_50_set1', 'FC_250_150_100_set1', 'FC_350_200_150_set1', 'FC_400_300_200_set1']

for i in range(len(fc_list)):
    # model building
    model = cnn_2_51_3class_v1(fc_list[i])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_logger = ModelLogger(model, model_name=model_name[i])

    # model training
    history = model.fit(x=train_x,
                        y=train_y,
                        batch_size=30,
                        epochs=300,
                        verbose=2,
                        validation_data=(test_x, test_y))

    model_logger.learning_curve(history=history, save=True, title=model_name[i])


# model_multiclass_evaluate(model, test_x=test_x, test_y=test_y)


