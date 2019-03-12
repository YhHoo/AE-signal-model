# this is for bash to know the path of the src
import sys
sys.path.append('C:/Users/YH/PycharmProjects/AE-signal-model')

import time
import tensorflow as tf
import argparse
from keras.optimizers import RMSprop
from src.experiment_dataset.dataset_experiment_2019_1_3 import AcousticEmissionDataSet
from src.model_bank.dataset_2018_7_13_lcp_recognition_model import *
from src.utils.helpers import *

# ------------------------------------------------------------------------------------------------------------ ARG PARSE
parser = argparse.ArgumentParser(description='Input some parameters.')
parser.add_argument('--model', default=1, type=str, help='Model Name')
parser.add_argument('--kernel_size', default=1, type=int, nargs='+', help='kernel size')
parser.add_argument('--fc_size', default=1, type=int, nargs='+', help='fully connected size')
parser.add_argument('--epoch', default=100, type=int, help='Number of training epoch')
parser.add_argument('--rmsprop_rho', default=100, type=float, help='Exponentially Weight Average over the square of gradient')
parser.add_argument('--l2_layer', default=0, type=int, help='layer to put l2 reg of 0.01')

args = parser.parse_args()
MODEL_SAVE_FILENAME = args.model
RESULT_SAVE_FILENAME = 'C:/Users/YH/PycharmProjects/AE-signal-model/result/{}_result.txt'.format(MODEL_SAVE_FILENAME)
KERNEL_SIZE = args.kernel_size
FC_SIZE = args.fc_size
EPOCH = args.epoch
RHO = args.rmsprop_rho
L2_LAYER = args.l2_layer

print('Model Name: ', MODEL_SAVE_FILENAME)
print('Result saving filename: ', RESULT_SAVE_FILENAME)
print('Conv Kernel size: ', KERNEL_SIZE)
print('FC neuron size: ', FC_SIZE)
print('rho of RMSProp: ', RHO)
print('Layer with l2 reg: ', L2_LAYER)

# ----------------------------------------------------------------------------------------------------------- GPU CONFIG
# instruct GPU to allocate only sufficient memory for this script
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# ------------------------------------------------------------------------------------------------------------ DATA PREP
ae_data = AcousticEmissionDataSet(drive='G')
train_x, train_y, test_x, test_y = ae_data.random_leak_bydist_downsampled_4(train_split=0.8)

train_x_reshape = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
test_x_reshape = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

train_y_cat = to_categorical(train_y, num_classes=6)
test_y_cat = to_categorical(test_y, num_classes=6)

# ------------------------------------------------------------------------------------------------------- MODEL TRAINING
lcp_model = LD_multiclass_model_4(kernel_size=KERNEL_SIZE, fc_size=FC_SIZE, l2_reg=L2_LAYER)
optimizer = RMSprop(lr=0.005, rho=RHO)
lcp_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

# saving best weight setting
logger = ModelLogger(model=lcp_model, model_name=MODEL_SAVE_FILENAME)
save_weight_checkpoint = logger.save_best_weight_cheakpoint(monitor='val_loss', period=5)

# start training
total_epoch = EPOCH
time_train_start = time.time()
history = lcp_model.fit(x=train_x_reshape,
                        y=train_y_cat,
                        validation_data=(test_x_reshape, test_y_cat),
                        callbacks=[save_weight_checkpoint],
                        epochs=total_epoch,
                        batch_size=200,
                        shuffle=True,
                        verbose=2)
time_train = time.time() - time_train_start

logger.save_architecture(save_readable=True)

# ------------------------------------------------------------------------------------------------------- LEARNING CURVE
# name for fig suptitle and filename
lr_name = '{}_LrCurve'.format(MODEL_SAVE_FILENAME)
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

# evaluate ------------------------------------------------------------------------------------------ EVALUATE REPORTING
# no of trainable parameter
trainable_count = int(np.sum([K.count_params(p) for p in set(lcp_model.trainable_weights)]))

# find highest val acc and lowest loss
best_val_acc_index = np.argmax(history.history['val_acc'])
best_val_loss_index = np.argmin(history.history['val_loss'])

# loading best model saved
lcp_best_model = load_model(model_name=MODEL_SAVE_FILENAME)

# test with val data
time_predict_start = time.time()
prediction = lcp_best_model.predict(test_x_reshape)
time_predict = time.time() - time_predict_start

prediction_argmax = np.argmax(prediction, axis=1)
actual_argmax = np.argmax(test_y_cat, axis=1)

# plot validation data
evaluate_name = '{}_Evaluate'.format(MODEL_SAVE_FILENAME)
fig_evaluate = plt.figure(figsize=(10, 7))
fig_evaluate.subplots_adjust(left=0.08, bottom=0.07, right=0.96, top=0.89)
fig_evaluate.suptitle(evaluate_name)
ax_evaluate = fig_evaluate.add_subplot(1, 1, 1)
ax_evaluate.plot(actual_argmax, color='r', label='Actual')
ax_evaluate.plot(prediction_argmax, color='b', label='Prediction', linestyle='None', marker='x')
ax_evaluate.legend()
fig_lr_save_filename = direct_to_dir(where='result') + '{}.png'.format(evaluate_name)
fig_evaluate.savefig(fig_lr_save_filename)

print('\n---------- EVALUATION RESULT SCRIPT LNL 1 -----------')
print('Model Trainable params: {}'.format(trainable_count))
print('Best Validation Accuracy: {:.4f} at Epoch {}/{}'.format(history.history['val_acc'][best_val_acc_index],
                                                               best_val_acc_index,
                                                               total_epoch))
print('Lowest Validation Loss: {:.4f} at Epoch {}/{}'.format(history.history['val_loss'][best_val_loss_index],
                                                             best_val_loss_index,
                                                             total_epoch))
print('Time taken to execute 1 sample: {}s'.format(time_predict / len(test_x_reshape)))
print('Time taken to complete {} epoch: {:.4f}s'.format(total_epoch, time_train))
rpf_result = logger.save_recall_precision_f1(y_pred=prediction_argmax, y_true=actual_argmax,
                                             all_class_label=[0, 1, 2, 3, 4, 5])


# saving the printed result again
with open(RESULT_SAVE_FILENAME, 'w') as f:
    f.write('\n---------- EVALUATION RESULT SCRIPT LNL 1 -----------')
    f.write('\nModel Conv Kernels Size: {}, FC Size: {}, L2_layer: {}'.format(KERNEL_SIZE, FC_SIZE, L2_LAYER))
    f.write('\nModel Trainable params: {}'.format(trainable_count))
    f.write('\nBest Validation Accuracy: {:.4f} at Epoch {}/{}'.format(history.history['val_acc'][best_val_acc_index],
                                                                       best_val_acc_index,
                                                                       total_epoch))
    f.write('\nLowest Validation Loss: {:.4f} at Epoch {}/{}'.format(history.history['val_loss'][best_val_loss_index],
                                                                     best_val_loss_index,
                                                                     total_epoch))
    f.write('\nTime taken to execute 1 sample: {}s'.format(time_predict / len(test_x_reshape)))
    f.write('\nTime taken to complete {} epoch: {:.4f}s'.format(total_epoch, time_train))

    for i in rpf_result:
        f.write('\n' + i)

    f.write('\n\nDist and Labels')
    f.write('\n[Leak @ 0m] -> class_0')
    f.write('\n[Leak @ 2m] -> class_1')
    f.write('\n[Leak @ 4m] -> class_2')
    f.write('\n[Leak @ 6m] -> class_3')
    f.write('\n[Leak @ 8m] -> class_4')
    f.write('\n[Leak @ 10m] -> class_5')