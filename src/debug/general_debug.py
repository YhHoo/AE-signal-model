import numpy as np
from random import shuffle
from scipy.signal import gausspulse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import signal
from scipy.signal import correlate as correlate_scipy
from numpy import correlate as correlate_numpy
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
# self lib
from src.controlled_dataset.ideal_dataset import white_noise
from src.utils.dsp_tools import spectrogram_scipy
from src.experiment_dataset.dataset_experiment_2018_5_30 import AcousticEmissionDataSet_30_5_2018

y_true = [0]*100 + [1]*100 + [2]*100
y_pred = [0]*30 + [1]*50 + [2]*20 + \
         [0]*20 + [1]*60 + [2]*20 + \
         [0]*10 + [1]*10 + [2]*80
# print(len(y_true))
# print(len(y_pred))
# data = confusion_matrix(y_true=y_true, y_pred=y_pred)

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


def recall_precision_multiclass(y_true, y_pred, all_class_label, verbose=True):
    # create labels for index and columns of confusion matrix
    col_labels = ['Actual_Class[{}]'.format(i) for i in all_class_label]
    index_labels = ['Predict_Class[{}]'.format(i) for i in all_class_label]

    # arrange all prediction and actual label into confusion matrix
    data = confusion_matrix(y_true=y_true, y_pred=y_pred)
    conf_mat = pd.DataFrame(data=data.T, index=index_labels, columns=col_labels)

    # taking all diagonals values into a 1d array
    diag = np.diag(conf_mat.values)

    # sum across rows and columns of confusion mat
    total_pred_of_each_class = pd.DataFrame.sum(conf_mat, axis=1).values
    total_samples_of_each_class = pd.DataFrame.sum(conf_mat, axis=0).values

    recall_each_class = diag / total_samples_of_each_class
    precision_each_class = diag / total_pred_of_each_class

    print('class recall: ', recall_each_class)
    print('class precision: ', precision_each_class)

    return conf_mat, recall_each_class, precision_each_class


class_label = np.arange(-1, 2, 1)
mat, r, p = recall_precision_multiclass(y_true, y_pred, all_class_label=class_label, verbose=True)

print(mat)
print(r)
print(p)




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


