import itertools
import numpy as np
import pywt
from multiprocessing import Pool
import gc
from random import shuffle
from scipy.signal import gausspulse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import signal
from scipy.signal import correlate as correlate_scipy
from numpy import correlate as correlate_numpy
import pandas as pd
from os import listdir
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
# self lib
from src.controlled_dataset.ideal_dataset import white_noise
from src.utils.dsp_tools import spectrogram_scipy, one_dim_xcor_2d_input
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.helpers import plot_heatmap_series_in_one_column, read_single_tdms, direct_to_dir, ProgressBarForLoop, \
                              break_into_train_test, ModelLogger, reshape_3d_to_4d_tocategorical
from src.model_bank.dataset_2018_7_13_leak_model import fc_leak_1bar_max_vec_v1

# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# ------------------------------------[TESTING MAX VEC GENERATOR]-------------------------------------------------------
# # wavelet
# m_wavelet = 'gaus1'
# scale = np.linspace(2, 30, 50)
# fs = 1e6
# # creating dict to store each class data
# all_class = {}
# for i in range(0, 11, 1):
#     all_class['class_[{}]'.format(i)] = []
#
#
# tdms_dir = direct_to_dir(where='yh_laptop_test_data') + '/1bar_leak/test_0001.tdms'
# # read raw from drive
# n_channel_data_near_leak = read_single_tdms(tdms_dir)
# n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)
#
# # split on time axis into no_of_segment
# n_channel_leak = np.split(n_channel_data_near_leak, axis=1, indices_or_sections=50)
#
# dist_diff = 0
# # for all sensor combination
# sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
#
# for sensor_pair in sensor_pair_near:
#     segment_no = 0
#     pb = ProgressBarForLoop(title='CWT+Xcor using {}'.format(sensor_pair), end=len(n_channel_leak))
#     # for all segmented signals
#     for segment in n_channel_leak:
#         pos1_leak_cwt, _ = pywt.cwt(segment[sensor_pair[0]], scales=scale, wavelet=m_wavelet,
#                                     sampling_period=1 / fs)
#         pos2_leak_cwt, _ = pywt.cwt(segment[sensor_pair[1]], scales=scale, wavelet=m_wavelet,
#                                     sampling_period=1 / fs)
#
#         # xcor for every pair of cwt
#         xcor, _ = one_dim_xcor_2d_input(input_mat=np.array([pos1_leak_cwt, pos2_leak_cwt]),
#                                         pair_list=[(0, 1)])
#         xcor = xcor[0]
#
#         # midpoint in xcor
#         mid = xcor.shape[1] // 2 + 1
#
#         max_xcor_vector = []
#         # for every row of xcor, find max point index
#
#         for row in xcor:
#             max_along_x = np.argmax(row)
#             max_xcor_vector.append(max_along_x - mid)
#         # store all feature vector for same class
#         all_class['class_[{}]'.format(dist_diff)].append(max_xcor_vector)
#
#         pb.update(now=segment_no)
#         segment_no += 1
#     dist_diff += 1
#     pb.destroy()
# # just to display the dict full dim
# l = []
# for _, value in all_class.items():
#     l.append(value)
# l = np.array(l)
# print(l.shape)
#
# # free up memory for unwanted variable
# pos1_leak_cwt, pos2_leak_cwt, n_channel_data_near_leak, l = None, None, None, None
# gc.collect()
#
# dataset = []
# label = []
# for i in range(0, 11, 1):
#     max_vec_list_of_each_class = all_class['class_[{}]'.format(i)]
#     dataset.append(max_vec_list_of_each_class)
#     label.append([i]*len(max_vec_list_of_each_class))
# dataset = np.concatenate(dataset, axis=0)
#
# # convert to array
# dataset = np.array(dataset)
# label = np.array(label)
# print('Dataset Dim: ', dataset.shape)
# print('Label Dim: ', label.shape)
#
# # save to csv
# label = label.reshape((-1, 1))
# all_in_one = np.concatenate([dataset, label], axis=1)
# column_label = ['Scale_{}'.format(i) for i in scale] + ['label']
# df = pd.DataFrame(all_in_one, columns=column_label)
# filename = direct_to_dir(where='result') + 'test.csv'
# df.to_csv(filename)

# ----------------------------------------------------------------------------------------------------------------------

# l = []
# for key, value in all_class.items():
#     l.append(value)
#
# l = np.array(l)
# print(l.shape)


# input = np.arange(300).reshape((10, 30))
#
# x = np.unravel_index(np.argmax(input, axis=None), input.shape)
# print(x)
# print(x[0])
# fig = plt.figure()
# cwt_ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
# colorbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.01])
# # title
# cwt_ax.set_title('Xcor of CWT')
# # plot
# cwt_ax.grid(linestyle='dotted')
# cwt_ax.axvline(x=input.shape[1] // 2 + 1, linestyle='dotted')
# cwt_ax.scatter(x[1], x[0], s=70, c='black', marker='x')
# cwt_ax.set_yticks([10, 50, 60, 70, 80, 82, 85, 89, 93, 99])
# i = cwt_ax.imshow(input, cmap='jet', aspect='auto', extent=[0, 30, 10, 100])
# plt.colorbar(i, cax=colorbar_ax, orientation='horizontal')
#
#
# plt.show()



# input = np.random.rand(3000).reshape((10, 300))
# act_1 = np.random.rand(2619).reshape((9, 291))
# act_2 = np.random.rand(2619).reshape((9, 291))
# act_3 = np.random.rand(2619).reshape((9, 291))
# act_4 = np.random.rand(2619).reshape((9, 291))
# act_5 = np.random.rand(2619).reshape((9, 291))
# act_6 = np.random.rand(2619).reshape((9, 291))
# act_7 = np.random.rand(2619).reshape((9, 291))
# act_8 = np.random.rand(2619).reshape((9, 291))
#
# act_11 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_21 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_31 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_41 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_51 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_61 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_71 = np.linspace(0, 1, 2619).reshape((9, 291))
# act_81 = np.linspace(0, 1, 2619).reshape((9, 291))
#
# val_test = [act_1, act_2, act_3, act_4, act_5, act_6, act_7, act_8]
# val_test_2 = [act_11, act_21, act_31, act_41, act_51, act_61, act_71, act_81]
# fig = plot_multiple_horizontal_heatmap(val_test, 'BIG TITLE', 'BIG TITLE')


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')


# col_labels = ['TargetLabel_class_1', 'TargetLabel_class_2', 'TargetLabel_class_3']
# index_labels = ['Predicted_class_1', 'Predicted_class_2', 'Predicted_class_3']
# conf_mat = pd.DataFrame(data=data.T, index=index_labels, columns=col_labels)
# # conf_mat['Total Prediction of Each Class'] = pd.DataFrame.sum(conf_mat, axis=1)
# diag = np.diag(conf_mat.values)
# total_pred_of_each_class = pd.DataFrame.sum(conf_mat, axis=1).values
# total_samples_of_each_class = pd.DataFrame.sum(conf_mat, axis=0).values
#
# recall_each_class = diag / total_samples_of_each_class
# precision_each_class = diag / total_pred_of_each_class
# print(conf_mat)
# print(diag)
# print(total_pred_of_each_class)
# print(total_samples_of_each_class)
# print('class recall: ', recall_each_class)
# print('class precision: ', precision_each_class)

# precision_c1 = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='micro')
# print(precision_c1)
# print(precision_c2)
# print(precision_c3)

# x = [1, 2, 3]
# dct = {}
# for i in x:
#     dct['lst_%s' % i] = []
#
# dct['lst_1'].append(np.arange(0, 10).reshape((2, 5)))
# dct['lst_1'].append(np.arange(5, 15).reshape((2, 5)))
# print(dct['lst_1'][0])


# x = [[i, i+1, i+2] for i in range(10)]
# print(x)
# shuffle(x)
# print(x)
#
#
# all_class = {}
# for i in range(-20, 21, 1):
#     all_class['class_[{}]'.format(i)] = []
#
# for i in range(5):
#     all_class['class_[0]'].append(np.linspace(0, i, 5))
#
# print(all_class['class_[0]'])
# l = np.array(all_class['class_[0]'])
# print(l)
# print(l.shape)

# shuffle(all_class['class_[0]'])
# # l = np.array(all_class['class_[0]'])
# # print(l)
#
# print(all_class['class_[0]'])
#
# all_class['class_[0]'] = all_class['class_[0]'][:2]
#
# print(all_class['class_[0]'])


# x = np.arange(0, 30, 1).reshape(2, 3, 5)
# x_shift = np.swapaxes(x, 1, 2)
# print(x)
# print(x_shift)

# fig = plt.figure(figsize=(5, 6))
# fig.suptitle('XCOR MAP of Leak Pos 1 & 2')
# ax1 = fig.add_axes([0.1, 0.51, 0.6, 0.39])  # [left, bottom, width, height]
# ax2 = fig.add_axes([0.1, 0.1, 0.6, 0.39])
# colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
# i = ax1.pcolormesh(np.arange(0, 600), np.arange(0, 41), values1)
# j = ax2.pcolormesh(np.arange(0, 600), np.arange(0, 41), values2)
# fig.colorbar(i, cax=colorbar_ax)
# fig.colorbar(j, cax=colorbar_ax)
# ax1.grid()
# ax2.grid()
# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


# -----------------------------------------------------------------------------
# fig = plt.figure(figsize=(5, 8))
# ax1 = fig.add_subplot(4, 1, 1)
# ax2 = fig.add_subplot(4, 1, 2)
# ax3 = fig.add_subplot(4, 1, 3)
# ax4 = fig.add_subplot(4, 1, 4)
# ax1.set_title('Signal 1')
# ax2.set_title('Signal 2')
# ax3.set_title('Xcor Signal [numpy]')
# ax4.set_title('Xcor Signal [scipy + fft]')
# ax1.plot(l)
# ax2.plot(m)
# ax3.plot(z)
# ax4.plot(z2)
# plt.subplots_adjust(hspace=0.6)
# plt.show()

# t = np.linspace(0, 10, 11)
# f = np.linspace(10, 100, 11)
# mat = np.arange(0, 100, 1).reshape((10, 10))
# print(t.shape)
# print(f.shape)
# print(mat.shape)
# print(mat)
# x_axis = np.arange(1, 11, 1)
# y_axis = np.arange(1, 11, 1)
#
# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
# colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
# i = ax.pcolormesh(x_axis, y_axis, mat)
# ax.grid()
# fig.colorbar(i, cax=colorbar_ax)
# ax.grid()
# ax.set_xlabel('Time [Sec]')
# ax.set_ylabel('Frequency [Hz]')
# ax.set_ylim(bottom=0, top=6, auto=True)
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax.set_title(plot_title)

# plt.show()

#
#
# assert m == item for i in l

# def colormapplot():
#     fig = plt.figure()
#     ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
#     colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
#     i = ax.pcolormesh(t, f, mat)
#     ax.grid()
#     fig.colorbar(i, cax=colorbar_ax)
#
#     return fig
#
#
# for i in range(3):
#     _ = colormapplot()
#     plt.close()
#
# fig1 = colormapplot()
# plt.show()


# three_dim_visualizer()
# data_3d = np.array([[[1],
#                      [3]],
#                     [[2],
#                      [4]],
#                     [[3],
#                      [-5]]])
# print(data_3d.min())
# print(data_3d.shape)
# print(data_3d.shape[1])
# print(data_3d[0].shape[0])

# s = np.arange(0, 100, 1).reshape((5, 20))
# x = np.arange(0, 20, 1)
# y = np.arange(0, 5, 1)
# print(s)
# print(x)
# print(y)
# mlb.barchart(y, x, s)
# mlb.imshow()

# sig = np.repeat([0., 1., 0., 1], 100)
# win = signal.hann(50)
# print(sum(win))
# mat1 = np.array([3, 9, 2, 1, 0, 0, 0, 0])
# mat2 = np.array([0, 3, 9, 2, 1, 0, 0, 0])
# plt.plot(sig)
# plt.plot(win, marker='x')
# plt.show()


# ori_signal = np.concatenate((mat1, mat3), axis=0)
# lag_signal = np.concatenate((mat2, mat4), axis=0)
# print(ori_signal.shape)


