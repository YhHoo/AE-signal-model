'''
THIS METHOD IS TO USE FULLY CONNECTED NN TO TRY SOLVING THE PLB LOCALIZATION PROBLLEM
'''
import time
from keras import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.helpers import *


data = AcousticEmissionDataSet_13_7_2018(drive='F')
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

# train_x_reshape = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
# test_x_reshape = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

# to categorical
train_y_cat = to_categorical(train_y, num_classes=num_classes)
test_y_cat = to_categorical(test_y, num_classes=num_classes)

print(train_x.shape)
print(test_x.shape)
print(train_y_cat.shape)
print(test_y_cat.shape)


# FC ------------------------------------------------------------------------------------------------------------------
# iterate several times to find best model with F1-score
for i in range(3):
    print('ITERATION ', i)
    model = Sequential()
    model.add(Dense(108, activation='relu', input_dim=3000))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(41, activation='softmax'))
    print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_logger = ModelLogger(model, model_name='PLB_2018_7_13_Classification_FC_take1')

    time_train_start = time.time()
    history = model.fit(x=train_x,
                        y=train_y_cat,
                        batch_size=100,
                        validation_data=(test_x, test_y_cat),
                        epochs=100,
                        verbose=2)
    time_train_taken = time.time() - time_train_start

    model_logger.learning_curve(history=history, save=True, show=False)
    model_logger.save_architecture(save_readable=True)

    # EVALUATE -------------------------------------------------------------------------------------------------------------
    # find best val acc
    best_val_acc_index = np.argmax(history.history['val_acc'])
    best_val_loss_index = np.argmin(history.history['val_loss'])

    # evaluate the model with all train and test data
    x = np.concatenate((train_x, test_x), axis=0)
    y = np.concatenate((train_y_cat, test_y_cat), axis=0)

    # execution time starts
    time_start = time.time()
    prediction = model.predict(x)
    time_taken = time.time() - time_start

    # reverse to_categorical
    prediction = np.argmax(prediction, axis=1)
    actual = np.argmax(y, axis=1)

    class_label = [i for i in range(0, 21, 1)] + [j for j in range(-20, 0, 1)]
    # save 2 csv of confusion matrix and recall_precision table

    print('\n---------- EVALUATION RESULT -----------')
    print('Total Training Time: {:.4f}s'.format(time_train_taken))
    print('Best Validation Accuracy: {:.4f} at Epoch {}'.format(history.history['val_acc'][best_val_acc_index],
                                                                best_val_acc_index))
    print('Lowest Validation Loss: {:.4f} at Epoch {}'.format(history.history['val_loss'][best_val_loss_index],
                                                              best_val_loss_index))
    print('Execution time for {} samples : {:.4f}s'.format(len(x), time_taken))
    model_logger.save_recall_precision_f1(y_true=actual, y_pred=prediction, all_class_label=class_label)