'''
THIS METHOD IS TO USE SVM TO TRY SOLVING THE PLB LOCALIZATION PROBLLEM
'''
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from keras import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.helpers import *


data = AcousticEmissionDataSet_13_7_2018(drive='E')
dataset, label = data.plb()

dataset_flatten = []
for xcor_map in dataset:
    dataset_flatten.append(xcor_map.ravel())

dataset_flatten = np.array(dataset_flatten)
print(dataset_flatten.shape)

# split to train test data
num_classes = 41
train_x, train_y, test_x, test_y = break_balanced_class_into_train_test(input=dataset_flatten,
                                                                        label=label,
                                                                        num_classes=num_classes,
                                                                        train_split=0.7,
                                                                        verbose=True)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

# SVM ------------------------------------------------------------------------------------------------------------------
model = svm.SVC(kernel='linear', C=3, gamma=1)
model.fit(train_x, train_y)
score = model.score(train_x, train_y)

predict = model.predict(test_x)
acc = accuracy_score(y_pred=predict, y_true=test_y)
print('SVM classification acc: ', acc)

# EVALUATE -------------------------------------------------------------------------------------------------------------
# evaluate the model with all train and test data
x = np.concatenate((train_x, test_x), axis=0)
y = np.concatenate((train_y, test_y), axis=0)
prediction = model.predict(x)
print('SVM output :', prediction)

class_label = [i for i in range(0, 21, 1)] + [j for j in range(-20, 0, 1)]

# SVM EVALUATE ---------------------------------------------------------------------------------------------------------
mat, r, p, f1 = compute_recall_precision_multiclass(y_true=y, y_pred=prediction, all_class_label=class_label)

mat_filename = 'SVM_confusion_mat.csv'
recall_precision_df_filename = 'SVM_recall_prec_f1.csv'

# prepare and save confusion matrix
mat.to_csv(mat_filename)

# prepare and save each class recall n precision n f1
mat = np.array([r, p])
recall_precision_df = pd.DataFrame(mat,
                                   index=['recall', 'precision'],
                                   columns=class_label)
recall_precision_df.loc['f1'] = None
recall_precision_df.iloc[2, 0] = f1
recall_precision_df.to_csv(recall_precision_df_filename)