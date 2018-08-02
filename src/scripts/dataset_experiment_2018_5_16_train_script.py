'''
This file trains the model on dataset_experiment_16_5_2018().
this model expecting the input data of 0.2 seconds with shape of (3000/1000/700, 40) only.
'''

from src.experiment_dataset.dataset_experiment_2018_5_16 import AccousticEmissionDataSet_16_5_2018
from keras.utils import to_categorical
from keras import optimizers
# self defined library
from src.utils.helpers import ModelLogger, evaluate_model_for_all_class
from src.model_bank.cnn_model_bank import cnn_1000_40_7class_v1, cnn_700_40_7class_v1

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

# optimizer
adam_optimizer = optimizers.Adam(lr=0.01)

# data summary
print('\n----------INPUT DATA DIMENSION---------')
print('Train_X dim: ', train_x.shape)
print('Train_Y dim: ', train_y.shape)
print('Test_X dim: ', test_x.shape)
print('Test_Y dim: ', test_y.shape)

# time-res = 5ms per band, f-res = 100Hz per band
model = cnn_1000_40_7class_v1()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# save model architecture
model_logger = ModelLogger(model, model_name='test1_CNN_22_5_18')
model_logger.save_architecture(save_readable=True)

# checkpoint
callback_list = model_logger.save_best_weight_cheakpoint(monitor='val_loss', period=1)

# Model Training
history = model.fit(x=train_x,
                    y=train_y,
                    validation_data=(test_x, test_y),
                    batch_size=30,
                    epochs=10,
                    shuffle=True,
                    callbacks=callback_list)

# save the learning curve
model_logger.learning_curve(history, save=True, title='F-range (0-100kHz)')

evaluate_model_for_all_class(model, test_x=test_x, test_y=test_y)

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

model = cnn_700_40_7class_v1()
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model logging
model_logger = ModelLogger(model, model_name='test2_CNN_23_5_18')
model_logger.save_architecture(save_readable=True)
callback_list = model_logger.save_best_weight_cheakpoint()

history = model.fit(x=train_x,
                    y=train_y,
                    validation_data=(test_x, test_y),
                    batch_size=30,
                    epochs=2,
                    shuffle=True,
                    callbacks=callback_list)

# learning curve
model_logger.learning_curve(history, save=True, title='F-range (0-70kHz)')

evaluate_model_for_all_class(model, test_x=test_x, test_y=test_y)
