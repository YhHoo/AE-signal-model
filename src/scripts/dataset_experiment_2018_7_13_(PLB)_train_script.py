'''
this script train the CNN on recognizing the PLB collected at different points
time series signal -> segmentation -> bandpass -> STFT -> Xcor -> preprocessing -> CNN training
'''
import numpy as np
import time
from keras.callbacks import TensorBoard
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.helpers import break_balanced_class_into_train_test, reshape_3d_to_4d_tocategorical, \
                              ModelLogger, compute_recall_precision_multiclass, evaluate_model_for_all_class
from src.model_bank.nn_model_bank import cnn2d_plb_v1


data = AcousticEmissionDataSet_13_7_2018(drive='F')
dataset, label = data.plb()

# split to train test data
num_classes = 41
train_x, train_y, test_x, test_y = break_balanced_class_into_train_test(input=dataset,
                                                                        label=label,
                                                                        num_classes=num_classes,
                                                                        train_split=0.7,
                                                                        verbose=True)

# reshape to satisfy conv2d input shape
train_x, train_y, test_x, test_y = reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y,
                                                                  fourth_dim=1,
                                                                  num_classes=num_classes,
                                                                  verbose=True)

# iterate several times to find best model with F1-score
for i in range(3):
    print('ITERATION ', i)
    model = cnn2d_plb_v1(input_shape=(train_x.shape[1], train_x.shape[2]), num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_logger = ModelLogger(model, model_name='PLB_2018_7_13_Classification_CNN[55k]_take4')
    # tensorboard (folder name in Graph will be used to name the run)
    # tb_callback = TensorBoard(log_dir='./Graph', histogram_freq=2,
    #                           write_graph=True, write_images=True, write_grads=True)
    sbw_callback = model_logger.save_best_weight_cheakpoint(monitor='val_acc', period=1)

    time_train_start = time.time()
    history = model.fit(x=train_x,
                        y=train_y,
                        batch_size=100,
                        validation_data=(test_x, test_y),
                        epochs=150,
                        verbose=2,
                        callbacks=[sbw_callback])
    time_train_taken = time.time() - time_train_start

    model_logger.learning_curve(history=history, save=True, show=False)
    model_logger.save_architecture(save_readable=True)

    # evaluate the model with all train and test data
    x = np.concatenate((train_x, test_x), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    time_start = time.time()
    prediction = model.predict(x)
    time_taken = time.time() - time_start

    prediction = np.argmax(prediction, axis=1)
    actual = np.argmax(y, axis=1)
    class_label = [i for i in range(0, 21, 1)] + [j for j in range(-20, 0, 1)]
    # save 2 csv of confusion matrix and recall_precision table
    model_logger.save_recall_precision_f1(y_true=actual, y_pred=prediction, all_class_label=class_label)

    # find best val acc
    best_val_acc_index = np.argmax(history.history['val_acc'])
    best_val_loss_index = np.argmin(history.history['val_loss'])

    print('\n---------- EVALUATION RESULT -----------')
    print('Total Training Time: {:.4f}s'.format(time_train_taken))
    print('Best Validation Accuracy: {:.4f} at Epoch {}'.format(history.history['val_acc'][best_val_acc_index],
                                                                best_val_acc_index))
    print('Lowest Validation Loss: {:.4f} at Epoch {}'.format(history.history['val_loss'][best_val_loss_index],
                                                              best_val_loss_index))
    print('Execution time for {} samples : {:.4f}s'.format(len(x), time_taken))
    model_logger.save_recall_precision_f1(y_true=actual, y_pred=prediction, all_class_label=class_label)



