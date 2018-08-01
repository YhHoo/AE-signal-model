import itertools
import numpy as np
from random import shuffle
from scipy.signal import gausspulse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import signal
from scipy.signal import correlate as correlate_scipy
from numpy import correlate as correlate_numpy
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
# self lib
from src.controlled_dataset.ideal_dataset import white_noise
from src.utils.dsp_tools import spectrogram_scipy
from src.experiment_dataset.dataset_experiment_2018_5_30 import AcousticEmissionDataSet_30_5_2018
from src.utils.helpers import plot_multiple_horizontal_heatmap


input = np.random.rand(3000).reshape((10, 300))
act_1 = np.random.rand(2619).reshape((9, 291))
act_2 = np.random.rand(2619).reshape((9, 291))
act_3 = np.random.rand(2619).reshape((9, 291))
act_4 = np.random.rand(2619).reshape((9, 291))
act_5 = np.random.rand(2619).reshape((9, 291))
act_6 = np.random.rand(2619).reshape((9, 291))
act_7 = np.random.rand(2619).reshape((9, 291))
act_8 = np.random.rand(2619).reshape((9, 291))

act_11 = np.linspace(0, 1, 2619).reshape((9, 291))
act_21 = np.linspace(0, 1, 2619).reshape((9, 291))
act_31 = np.linspace(0, 1, 2619).reshape((9, 291))
act_41 = np.linspace(0, 1, 2619).reshape((9, 291))
act_51 = np.linspace(0, 1, 2619).reshape((9, 291))
act_61 = np.linspace(0, 1, 2619).reshape((9, 291))
act_71 = np.linspace(0, 1, 2619).reshape((9, 291))
act_81 = np.linspace(0, 1, 2619).reshape((9, 291))

val_test = [act_1, act_2, act_3, act_4, act_5, act_6, act_7, act_8]
val_test_2 = [act_11, act_21, act_31, act_41, act_51, act_61, act_71, act_81]
# fig = plot_multiple_horizontal_heatmap(val_test, 'BIG TITLE', 'BIG TITLE')

fig = plt.figure(figsize=(15, 7))
fig.subplots_adjust(left=0.06, right=0.96)
# main title of figure
fig.suptitle('Conv2d_1 Layer Activation')
# all axes grid's big title
fig.text(0.10, 0.9, 'TESTING')
fig.text(0.35, 0.9, 'TESTING')
fig.text(0.58, 0.9, 'TESTING')
fig.text(0.82, 0.9, 'TESTING')

grid_0 = AxesGrid(fig, 141,
                  nrows_ncols=(1, 1),
                  axes_pad=0.1,
                  share_all=True,
                  label_mode="L",
                  cbar_location="bottom",
                  cbar_mode="single",
                  cbar_size='15%')

grid_1 = AxesGrid(fig, 142,
                  nrows_ncols=(8, 1),
                  axes_pad=0.1,
                  share_all=True,
                  label_mode="L",
                  cbar_location="right",
                  cbar_mode="single",
                  cbar_size='0.5%')
grid_2 = AxesGrid(fig, 143,
                  nrows_ncols=(8, 1),
                  axes_pad=0.1,
                  share_all=True,
                  label_mode="L",
                  cbar_location="right",
                  cbar_mode="single",
                  cbar_size='0.5%')
grid_3 = AxesGrid(fig, 144,
                  nrows_ncols=(8, 1),
                  axes_pad=0.1,
                  share_all=True,
                  label_mode="L",
                  cbar_location="right",
                  cbar_mode="single",
                  cbar_size='0.5%')

for ax in grid_0:
    im = ax.imshow(input, vmin=0, vmax=1, extent=(0.01, 0.91, 0.6, 0.39), cmap='jet')

for val, ax in zip(val_test, grid_1):
    # this configure titles for each heat map
    ax.set_title('TEST', position=(-0.15, 0.388), fontsize=7, rotation='vertical')
    # this configure the dimension of the heat map in the fig object
    im = ax.imshow(val, vmin=0, vmax=1, extent=(0.01, 0.91, 0.6, 0.39), cmap='jet')  # (left, right, bottom, top)

for val, ax in zip(val_test_2, grid_2):
    # this configure titles for each heat map
    ax.set_title('TEST', position=(-0.15, 0.388), fontsize=7, rotation='vertical')
    # this configure the dimension of the heat map in the fig object
    im = ax.imshow(val, vmin=0, vmax=1, extent=(0.01, 0.91, 0.6, 0.39), cmap='jet')  # (left, right, bottom, top)

for val, ax in zip(val_test, grid_3):
    # this configure titles for each heat map
    ax.set_title('TEST', position=(-0.15, 0.388), fontsize=7, rotation='vertical')
    # this configure the dimension of the heat map in the fig object
    im = ax.imshow(val, vmin=0, vmax=1, extent=(0.01, 0.91, 0.6, 0.39), cmap='jet')  # (left, right, bottom, top)

# this simply add color bar instance
grid_0.cbar_axes[0].colorbar(im)
grid_1.cbar_axes[0].colorbar(im)
grid_2.cbar_axes[0].colorbar(im)
grid_3.cbar_axes[0].colorbar(im)

# this toggle labels for color bar
for cax in grid_0.cbar_axes:
    cax.toggle_label(True)
for cax in grid_1.cbar_axes:
    cax.toggle_label(True)
for cax in grid_2.cbar_axes:
    cax.toggle_label(True)
for cax in grid_3.cbar_axes:
    cax.toggle_label(True)

plt.show()


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


