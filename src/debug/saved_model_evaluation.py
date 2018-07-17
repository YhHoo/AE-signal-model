'''
This code laod the saved model in .h5 and .json, use predict() to try classifying new/test data
'''
import matplotlib.pyplot as plt
import numpy as np
# self defined library
from src.utils.helpers import model_loader, model_multiclass_evaluate, break_into_train_test, \
                              reshape_3d_to_4d_tocategorical, three_dim_visualizer
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018

# -------------------[LOADING DATA]----------------------------
# data set
ae_data = AcousticEmissionDataSet_13_7_2018(drive='F')
dataset = ae_data.plb_unseen()
dataset = dataset.reshape((dataset.shape[0], dataset.shape[1], dataset.shape[2], 1))

# split to train test data
# num_classes = 6
# train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
#                                                          label=label,
#                                                          num_classes=num_classes,
#                                                          train_split=0.5,
#                                                          verbose=True)
#
# # reshape to satisfy conv2d input shape
# train_x, train_y, test_x, test_y = reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y,
#                                                                   fourth_dim=1,
#                                                                   num_classes=num_classes,
#                                                                   verbose=True)

# -------------------[LOADING MODEL]----------------------------

model = model_loader(model_name='PLB_2018_7_13_Classification_CNN[9.7M]')
model.compile(loss='categorical_crossentropy', optimizer='adam')

prediction = model.predict(dataset)
print(prediction)
fig = three_dim_visualizer(x_axis=np.arange(1, 7, 1),
                           y_axis=np.arange(0, prediction.shape[0], 1),
                           zxx=prediction,
                           label=['class output', 'samples', 'probability'],
                           output='3d',
                           title='Test the PLB_2018_7_13_Classification_CNN[9.7M] on Unseen 3m')
plt.show()

# print('------------MODEL EVALUATION-------------')
# model_multiclass_evaluate(model, test_x=test_x, test_y=test_y)
# model_multiclass_evaluate(model, test_x=train_x, test_y=train_y)



