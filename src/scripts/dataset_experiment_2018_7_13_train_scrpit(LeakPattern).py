from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
# self lib
from src.utils.helpers import direct_to_dir, break_into_train_test, ModelLogger
from src.model_bank.dataset_2018_7_13_leak_model import fc_leak_1bar_max_vec_v2, fc_leak_1bar_max_vec_v1
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018

# Config
num_classes = 11
nn_input_shape = (81, )

# reading data ---------------------------------------------------------------------------------------------------------
data = AcousticEmissionDataSet_13_7_2018(drive='F')
dataset, label = data.leak_1bar_in_cwt_xcor_maxpoints_vector(dataset_no=2,
                                                             f_range_to_keep=(18, 99),
                                                             class_to_keep='all')

# data pre-processing
train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=num_classes,
                                                         train_split=0.7,
                                                         verbose=True,
                                                         shuffled_each_class=True)

# to categorical
train_y_cat = to_categorical(train_y, num_classes=num_classes)
test_y_cat = to_categorical(test_y, num_classes=num_classes)


# training -------------------------------------------------------------------------------------------------------------
model = fc_leak_1bar_max_vec_v2(input_shape=nn_input_shape, num_classes=num_classes)
# optimizer
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
model_logger = ModelLogger(model, model_name='fc_leak_1bar_max_vec_v2')
history = model.fit(x=train_x,
                    y=train_y_cat,
                    batch_size=500,
                    validation_data=(test_x, test_y_cat),
                    epochs=100,
                    verbose=2)
model_logger.learning_curve(history=history, save=False, show=True)

# def create_model():
#     model = fc_leak_1bar_max_vec_v2(input_shape=nn_input_shape, num_classes=num_classes)
#     # optimizer
#     optimizer = Adam(lr=0.001)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
#
#     return model
#
#
# model = KerasClassifier(build_fn=create_model(), verbose=0, epochs=100)
#
# # grid search param
# batch_size = [200, 500]  # , 600, 700, 800, 900, 1000, 2000
# param_grid = dict(batch_size=batch_size)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
# grid_result = grid.fit(train_x, train_y)
#
# # model_logger = ModelLogger(model, model_name='fc_leak_1bar_max_vec_v2')
# # history = model.fit(x=train_x,
# #                     y=train_y_cat,
# #                     batch_size=500,
# #                     validation_data=(test_x, test_y_cat),
# #                     epochs=60,
# #                     verbose=2)
# # model_logger.learning_curve(history=history, save=False, show=True)
#
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))