import numpy as np
# self declared library
from ideal_dataset import noise_time_shift_dataset
from utils import break_into_train_test, ModelLogger, model_multiclass_evaluate, reshape_3d_to_4d_tocategorical
from cnn_model_bank import fc_2x51_3class


# time axis setting
fs = 1000
duration = 20  # tune this for duration
total_point = int(fs * duration)
time_axis = np.linspace(0, duration, total_point)

dataset, label = noise_time_shift_dataset(time_axis, fs=fs, verbose=True, num_series=10)

train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=3,
                                                         train_split=0.7,
                                                         verbose=True)

# convert labels to categorical only
_, train_y, _, test_y = reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y,
                                                       fourth_dim=1, num_classes=3, verbose=True)
train_x = np.reshape(train_x, (train_x.shape[0], 102))
test_x = np.reshape(test_x, (test_x.shape[0], 102))

fc = [1500, 1000, 800, 400, 100]
model = fc_2x51_3class(fc)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_logger = ModelLogger(model, model_name='FC_2x51')

# model training
history = model.fit(x=train_x,
                    y=train_y,
                    batch_size=1000,
                    epochs=300,
                    verbose=2,
                    validation_data=(test_x, test_y))

model_logger.learning_curve(history=history, save=True, title='FC_2x51')
