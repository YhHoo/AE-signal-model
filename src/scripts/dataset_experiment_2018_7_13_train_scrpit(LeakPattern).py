from keras.utils import to_categorical
from src.utils.helpers import direct_to_dir, break_into_train_test, ModelLogger
from src.model_bank.dataset_2018_7_13_leak_model import fc_leak_1bar_max_vec_v1
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018

# Config
num_classes = 11
nn_input_shape = (50, )

# reading data
data = AcousticEmissionDataSet_13_7_2018(drive='F')
dataset, label = data.leak_1bar_in_cwt_xcor_maxpoints_vector()

# data pre-processing
train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=num_classes,
                                                         train_split=0.8,
                                                         verbose=True,
                                                         shuffled_each_class=True)

# to categorical
train_y_cat = to_categorical(train_y, num_classes=num_classes)
test_y_cat = to_categorical(test_y, num_classes=num_classes)

# training
model = fc_leak_1bar_max_vec_v1(input_shape=nn_input_shape, num_classes=num_classes)
model_logger = ModelLogger(model, model_name='fc_leak_1bar_max_vec_v1')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x=train_x,
                    y=train_y_cat,
                    batch_size=200,
                    validation_data=(test_x, test_y_cat),
                    epochs=200,
                    verbose=2)
model_logger.learning_curve(history=history, save=False, show=True)

