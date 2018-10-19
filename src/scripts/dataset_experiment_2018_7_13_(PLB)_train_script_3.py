'''
THIS METHOD IS TO USE FULLY CONNECTED NN TO TRY SOLVING THE PLB LOCALIZATION PROBLLEM
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

train_x_reshape = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
test_x_reshape = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

# to categorical
train_y_cat = to_categorical(train_y, num_classes=num_classes)
test_y_cat = to_categorical(test_y, num_classes=num_classes)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)


# FC ------------------------------------------------------------------------------------------------------------------
model = Sequential()
model.add(Dense(108, activation='relu', input_dim=3000))
model.add(Dense(64, activation='relu'))
# model.add(Dense)
model.add(Dense(41, activation='softmax'))
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_logger = ModelLogger(model, model_name='PLB_2018_7_13_Classification_FC_take1')

history = model.fit(x=train_x,
                    y=train_y_cat,
                    batch_size=100,
                    validation_data=(test_x, test_y_cat),
                    epochs=100,
                    verbose=2)

model_logger.learning_curve(history=history, save=True, show=False)
model_logger.save_architecture(save_readable=True)

# EVALUATE -------------------------------------------------------------------------------------------------------------
# evaluate the model with all train and test data
x = np.concatenate((train_x, test_x), axis=0)
y = np.concatenate((train_y, test_y), axis=0)
prediction = model.predict(x)
print('SVM output :', prediction)
prediction = np.argmax(prediction, axis=1)
actual = np.argmax(y, axis=1)
class_label = [i for i in range(0, 21, 1)] + [j for j in range(-20, 0, 1)]
# save 2 csv of confusion matrix and recall_precision table
model_logger.save_recall_precision_f1(y_true=actual, y_pred=prediction, all_class_label=class_label)


