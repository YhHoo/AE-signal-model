'''
this script train the CNN on recognizing the signal of 2 different leak position, differ by 1m
time series signal -> segmentation -> bandpass -> STFT -> Xcor -> preprocessing -> CNN training
'''

from src.experiment_dataset.dataset_experiment_30_5_2018 import AcousticEmissionDataSet_30_5_2018
from src.utils.helpers import break_into_train_test, reshape_3d_to_4d_tocategorical, \
                              ModelLogger, model_multiclass_evaluate
from src.model_bank.cnn_model_bank import cnn_general_v1

data = AcousticEmissionDataSet_30_5_2018(drive='F')
dataset, label = data.leak_2class()

# train test data packaging
num_classes = 2
train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=num_classes,
                                                         train_split=0.7,
                                                         verbose=True)

# reshape to satisfy conv2d input shape
train_x, train_y, test_x, test_y = reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y,
                                                                  fourth_dim=1, num_classes=num_classes, verbose=True)

model = cnn_general_v1(input_shape=(train_x.shape[1], train_x.shape[2]), num_classes=num_classes)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_logger = ModelLogger(model, model_name='cnn_general_v1')
history = model.fit(x=train_x,
                    y=train_y,
                    batch_size=30,
                    validation_data=(test_x, test_y),
                    epochs=150,
                    verbose=1)

model_logger.learning_curve(history=history, save=False, show=True)
model_multiclass_evaluate(model, test_x=test_x, test_y=test_y)