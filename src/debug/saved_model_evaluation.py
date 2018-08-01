'''
This code laod the saved model in .h5 and .json, use predict() to try classifying new/test data
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
# self defined library
from src.utils.helpers import model_loader, model_multiclass_evaluate, break_into_train_test, \
                              reshape_3d_to_4d_tocategorical, three_dim_visualizer, compute_recall_precision_multiclass
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018

# -------------------[LOADING DATA]----------------------------
data = AcousticEmissionDataSet_13_7_2018(drive='F')
dataset, label = data.plb()

# split to train test data
num_classes = 41
train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=num_classes,
                                                         train_split=0.7,
                                                         verbose=True)
# reshape to satisfy conv2d input shape
train_x, train_y, test_x, test_y = reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y,
                                                                  fourth_dim=1,
                                                                  num_classes=num_classes,
                                                                  verbose=True)
x = np.concatenate((train_x, test_x), axis=0)
y = np.concatenate((train_y, test_y), axis=0)

print(x.shape)
print(y.shape)
# -------------------[LOADING MODEL]----------------------------

model = model_loader(model_name='PLB_2018_7_13_Classification_CNN[33k]_take2')
model.compile(loss='categorical_crossentropy', optimizer='adam')
x = np.concatenate((train_x, test_x), axis=0)
y = np.concatenate((train_y, test_y), axis=0)
prediction = model.predict(x)
prediction = np.argmax(prediction, axis=1)
actual = np.argmax(y, axis=1)
print(prediction)
print(actual)
# fig = three_dim_visualizer(x_axis=np.arange(1, 7, 1),
#                            y_axis=np.arange(0, prediction.shape[0], 1),
#                            zxx=prediction,
#                            label=['class output', 'samples', 'probability'],
#                            output='3d',
#                            title='Test the PLB_2018_7_13_Classification_CNN[9.7M] on Unseen 3m')
# plt.show()

# print('------------MODEL EVALUATION-------------')
model_multiclass_evaluate(model, test_x=test_x, test_y=test_y)
class_label = [i for i in range(0, 21, 1)] + [j for j in range(-20, 0, 1)]
mat, r, p = compute_recall_precision_multiclass(y_true=actual, y_pred=prediction, all_class_label=class_label, verbose=True)

mat.to_csv('PLB_2018_7_13_Classification_CNN[33k]_confusion_mat.csv')
print(r)
print(p)








