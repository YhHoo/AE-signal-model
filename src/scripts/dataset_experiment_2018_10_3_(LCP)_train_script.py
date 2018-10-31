import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
import gc
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from src.experiment_dataset.dataset_experiment_2018_10_3 import AcousticEmissionDataSet_3_10_2018
from src.model_bank.dataset_2018_7_13_lcp_recognition_model import *
from src.utils.helpers import *

# instruct GPU to allocate only sufficient memory for this script
config = tf.ConfigProto()
config.gpu_option.allow_growth = True
sess = tf.Session(config=config)

# repeat the training process for 3 times (take the best score)
# every iter, the train and test set will contain different combination of data because of the shuffle bfore split
for iter_no in range(3):
    ae_data = AcousticEmissionDataSet_3_10_2018(drive='F')
    train_x, train_y, test_x, test_y = ae_data.lcp_dataset_binary_class(train_split=0.7)

    train_x_reshape = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    test_x_reshape = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

    lcp_model = lcp_recognition_binary_model_2()
    lcp_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    # saving best weight setting
    logger = ModelLogger(model=lcp_model, model_name='LCP_Recog_1')
    save_weight_checkpoint = logger.save_best_weight_cheakpoint(monitor='val_loss', period=5)

    # tensorboard
    # tb_save_dir = direct_to_dir(where='result') + 'Graph/run_6, model1'
    # tb_callback = TensorBoard(log_dir=tb_save_dir,
    #                           histogram_freq=10,
    #                           write_graph=False,
    #                           write_images=False,
    #                           write_grads=True)

    # start training
    total_epoch = 1500
    time_train_start = time.time()
    history = lcp_model.fit(x=train_x_reshape,
                            y=train_y,
                            validation_data=(test_x_reshape, test_y),
                            callbacks=[save_weight_checkpoint],
                            epochs=total_epoch,
                            batch_size=350,
                            shuffle=True,
                            verbose=2)
    time_train = time.time() - time_train_start

    logger.save_architecture(save_readable=True)

    # plotting the learning curve
    # name for fig suptitle and filename
    lr_name = 'LrCurve_RUN_{}'.format(iter_no)
    fig_lr = plt.figure(figsize=(10, 7))
    fig_lr.subplots_adjust(left=0.08, bottom=0.07, right=0.96, top=0.89)
    fig_lr.suptitle(lr_name)
    ax_lr = fig_lr.add_subplot(1, 1, 1)
    ax_lr.plot(history.history['loss'], label='train_loss')
    ax_lr.plot(history.history['val_loss'], label='val_loss')
    ax_lr.plot(history.history['acc'], label='train_acc')
    ax_lr.plot(history.history['val_acc'], label='val_acc')
    ax_lr.legend()
    fig_lr_save_filename = direct_to_dir(where='result') + '{}.png'.format(lr_name)
    fig_lr.savefig(fig_lr_save_filename)

    # no of trainable parameter
    trainable_count = int(np.sum([K.count_params(p) for p in set(lcp_model.trainable_weights)]))

    # evaluate ---------------------------------------------------------------------------------------------------------

    # find highest val acc and lowest loss
    best_val_acc_index = np.argmax(history.history['val_acc'])
    best_val_loss_index = np.argmin(history.history['val_loss'])

    # loading best model saved
    lcp_best_model = load_model(model_name='LCP_Recog_1')

    # test with val data
    time_predict_start = time.time()
    prediction = lcp_best_model.predict(test_x_reshape)
    time_predict = time.time() - time_predict_start

    prediction_quantized = []
    for p in prediction:
        if p > 0.5:
            prediction_quantized.append(1)
        else:
            prediction_quantized.append(0)

    # plot validation data
    evaluate_name = 'Evaluate_ITER_{}'.format(iter_no)
    fig_evaluate = plt.figure(figsize=(10, 7))
    fig_evaluate.subplots_adjust(left=0.08, bottom=0.07, right=0.96, top=0.89)
    fig_evaluate.suptitle(evaluate_name)
    ax_evaluate = fig_evaluate.add_subplot(1, 1, 1)
    ax_evaluate.plot(test_y, color='r', label='Actual')
    ax_evaluate.plot(prediction, color='b', label='Prediction', linestyle='None', marker='x')
    ax_evaluate.legend()
    fig_lr_save_filename = direct_to_dir(where='result') + '{}.png'.format(evaluate_name)
    fig_evaluate.savefig(fig_lr_save_filename)

    print('\n---------- EVALUATION RESULT RUN_{} -----------'.format(iter_no))
    print('**Param in tuning --> [optimizer: rmsprop]')
    print('Model Trainable params: {}'.format(trainable_count))
    print('Best Validation Accuracy: {:.4f} at Epoch {}/{}'.format(history.history['val_acc'][best_val_acc_index],
                                                                   best_val_acc_index,
                                                                   total_epoch))
    print('Lowest Validation Loss: {:.4f} at Epoch {}/{}'.format(history.history['val_loss'][best_val_loss_index],
                                                                 best_val_loss_index,
                                                                 total_epoch))
    print('Time taken to execute 1 sample: {}s'.format(time_predict / len(test_x_reshape)))
    print('Time taken to complete {} epoch: {:.4f}s'.format(total_epoch, time_train))
    logger.save_recall_precision_f1(y_pred=prediction_quantized, y_true=test_y, all_class_label=[0, 1])

    gc.collect()
