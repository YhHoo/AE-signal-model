import numpy as np
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import LabelBinarizer
# self lib
from src.utils.helpers import direct_to_dir, break_into_train_test, ModelLogger
from src.model_bank.dataset_2018_7_13_leak_localize_model import fc_leak_1bar_max_vec_v2, fc_leak_1bar_max_vec_v1
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018

# Config
num_classes = 5
nn_input_shape = (100, )

# reading data ---------------------------------------------------------------------------------------------------------
data = AcousticEmissionDataSet_13_7_2018(drive='F')
dataset, label = data.leak_1bar_in_cwt_xcor_maxpoints_vector(dataset_name='bounded_xcor',
                                                             f_range_to_keep=(0, 100),
                                                             class_to_keep=[1, 3, 5, 7, 9],
                                                             shuffle=False)

# train_y_cat = to_categorical(train_y, num_classes=num_classes)
# test_y_cat = to_categorical(test_y, num_classes=num_classes)

# print(train_y_cat.shape)

# train using Stratified K-fold cross ----------------------------------------------------------------------------------
# skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)
# print('\nUsing Sklearn.model_selecton.StratifiedKfold()')
# for train, test in skf.split(X=dataset, y=label):
#     # to categorical
#     train_x = dataset[train]
#     train_y = label[train]
#     test_x = dataset[test]
#     test_y = label[test]
#     print(train_y.shape)
#     train_y_cat = to_categorical(train_y, num_classes=num_classes)
#     test_y_cat = to_categorical(test_y, num_classes=num_classes)
#     print(train_y_cat.shape)
#     model = fc_leak_1bar_max_vec_v2(input_shape=nn_input_shape, num_classes=num_classes)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
#     model_logger = ModelLogger(model, model_name='fc_leak_1bar_max_vec_v2')
#     history = model.fit(x=train_x,
#                         y=train_y_cat,
#                         batch_size=600,
#                         validation_data=(test_x, test_y_cat),
#                         epochs=100,
#                         verbose=0)
#     model_logger.learning_curve(history=history, save=False, show=True)
#     score = model.evaluate(x=test_x, y=test_y_cat)
#     print('Acc: ', score)

# normal training ------------------------------------------------------------------------------------------------------

# data pre-processing
train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=num_classes,
                                                         train_split=0.8,
                                                         verbose=True,
                                                         shuffled_each_class=True)

# label binarizer cn even works for jumping labels, e.g. [0, 0, 2, 4, 6, 8]
lb_encoder = LabelBinarizer()
train_y_cat = lb_encoder.fit_transform(train_y)
test_y_cat = lb_encoder.fit_transform(test_y)

model = fc_leak_1bar_max_vec_v2(input_shape=nn_input_shape, num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model_logger = ModelLogger(model, model_name='fc_leak_1bar_max_vec_v2')
history = model.fit(x=train_x,
                    y=train_y_cat,
                    batch_size=600,
                    validation_data=(test_x, test_y_cat),
                    epochs=300,
                    verbose=2)
model_logger.learning_curve(history=history, save=False, show=True)

# Grid search hyperparam -----------------------------------------------------------------------------------------------


# def create_model(optimizer='adam'):
#     model = fc_leak_1bar_max_vec_v2(input_shape=nn_input_shape, num_classes=num_classes)
#     # optimizer
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
#     return model
#
#
# seed = 7
# np.random.seed(seed)
# model = KerasClassifier(build_fn=create_model, verbose=0, epochs=100)
#
# # grid search param
# batch_size = [200, 500, 600, 700, 800, 1000, 2000]
# optimizer = ['SGD', 'adam', 'RMSprop']
# # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# param_grid = dict(batch_size=batch_size, optimizer=optimizer)
# grid = GridSearchCV(model,
#                     param_grid=param_grid,
#                     n_jobs=1,
#                     return_train_score=True,
#                     cv=3,
#                     verbose=1)
# grid_result = grid.fit(dataset, label)
#
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("Test_score: {} with std_dev: ({}) with: {}".format(mean, stdev, param))

