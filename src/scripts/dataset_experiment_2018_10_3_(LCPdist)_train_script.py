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
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# repeat the training process for 3 times (take the best score)
# every iter, the train and test set will contain different combination of data because of the shuffle bfore split
for iter_no in range(3):
    ae_data = AcousticEmissionDataSet_3_10_2018(drive='F')
    train_x, train_y, test_x, test_y = ae_data.lcp_by_distance_dataset_multi_class(train_split=0.7)

    train_x_reshape = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    test_x_reshape = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

    train_y_cat = to_categorical(train_y, num_classes=6)
    test_y_cat = to_categorical(test_y, num_classes=6)

    lcp_model = lcp_by_dist_recognition_multi_model_1()
    lcp_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    # saving best weight setting
    logger = ModelLogger(model=lcp_model, model_name='LCP_Dist_Recog_1')
    save_weight_checkpoint = logger.save_best_weight_cheakpoint(monitor='val_loss', period=5)

    # start training
    total_epoch = 1500
    time_train_start = time.time()
    history = lcp_model.fit(x=train_x_reshape,
                            y=train_y_cat,
                            validation_data=(test_x_reshape, test_y_cat),
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
