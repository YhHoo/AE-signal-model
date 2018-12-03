'''
THIS SCRIPT IS TO FEED THE LCP RECOGNITION MODEL WITH RAW AE DATA, AND SEE WHETHER IT WILL RECORGNIZE IT WELL. MODEL
WILL SLIDE THROUGH ONE 5M POINT AE RAW DATA, USING A WINDOW, WITH A STRIDE.
DATASET: UNSEEN DATASET FROM 13/7/18.
'''

import gc
import time
import matplotlib.patches as mpatches
import tensorflow as tf
from src.utils.helpers import *


# instruct GPU to allocate only sufficient memory for this script
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# SLIDING WINDOW CONFIG
window_stride = 10
window_size = (1000, 5000)
sample_size_for_prediction = 10000

# SAVING CONFIG
df_pred_save_filename = direct_to_dir(where='result') + 'pred_result_(noleak)2bar_near_test_0020.csv'

# file reading
# near -3,-2,2,4,6,8,10,12
noleak_1bar_near = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/No_Leak/test_0020.tdms'
leak_1bar_near = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/Leak/test_0010.tdms'
noleak_2bar_near = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/2 bar/No_Leak/test_0020.tdms'
leak_2bar_near = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/2 bar/Leak/test_0010.tdms'

# far -3,-2,10,14,16,18,20,22
noleak_1bar_far = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/1 bar/No_Leak/'
leak_1bar_far = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/1 bar/Leak/'
noleak_2bar_far = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/2 bar/No_Leak/'
leak_2bar_far = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/1 bar/Leak/'

# test for near
n_channel_data = read_single_tdms(noleak_2bar_near)
n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1]  # drop ch@12m
print('TDMS data dim: ', n_channel_data.shape)
total_len = n_channel_data.shape[1]
total_ch = len(n_channel_data)

# ensure enough length of data input
assert total_len > (window_size[0] + window_size[1]), 'Data length is too short, mz be at least {}'.\
    format(window_size[0] + window_size[1])
window_index = np.arange(window_size[0], (total_len - window_size[1]), window_stride)
print('Window Index Len: ', len(window_index))
print('Window Index: ', window_index)

# LOADING AND EXECUTE MODEL --------------------------------------------------------------------------------------------
# lcp_model = load_model(model_name='LCP_Dist_Recog_2x')
# lcp_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# print(lcp_model.summary())
#
# # creating index of sliding windows [index-1000, index+5000]
# scaler = MinMaxScaler(feature_range=(-1, 1))
#
# prediction_all_ch = []
# # for all channels
# for ch_no in range(total_ch):
#     temp, model_pred = [], []
#     pb = ProgressBarForLoop(title='Iterating all Samples in ch[{}]'.format(ch_no), end=len(window_index))
#     progress = 0
#     for index in window_index:
#         pb.update(now=progress)
#         data = n_channel_data[ch_no, (index - window_size[0]):(index + window_size[1])]
#         data_norm = scaler.fit_transform(data.reshape(-1, 1)).ravel()
#         temp.append(data_norm)
#
#         # detect for last entry
#         if progress < (len(window_index) - 1):
#
#             if len(temp) < sample_size_for_prediction:
#                 progress += 1
#                 continue
#             else:
#                 progress += 1
#                 # print('temp full !')
#
#         # do tis when temp is full
#         # reshape
#         temp = np.array(temp)
#         temp = temp.reshape((temp.shape[0], temp.shape[1], 1))
#         # print(temp.shape)
#         time_predict_start = time.time()
#         prediction = np.argmax(lcp_model.predict(temp), axis=1)
#
#         # # estimation of dist using all class posterior probability
#         # estimation = []
#         # for p in lcp_model.predict(temp):
#         #     estimation.append(p[0]*0 + p[1]*2 + p[2]*4.5 + p[3]*5 + p[4]*8 + p[5]*10)
#
#         time_predict = time.time() - time_predict_start
#         model_pred.append(prediction)
#         # reset temp
#         temp = []
#         # free up memory
#         gc.collect()
#
#     pb.destroy()
#     model_pred = np.concatenate(model_pred, axis=0)
#     print('Model Prediction Dim: ', model_pred.shape)
#
#     prediction_all_ch.append(model_pred)
#
# prediction_all_ch = np.array(prediction_all_ch).T
# df_pred = pd.DataFrame(data=prediction_all_ch,
#                        columns=['ch0[3m]', 'ch1[2m]', 'ch2[2m]', 'ch3[4m]', 'ch4[6m]', 'ch5[8m]', 'ch6[10m]'])
# df_pred.to_csv(df_pred_save_filename)
#
# print('Saved --> ', df_pred_save_filename)

# RESULT VISUALIZATION -------------------------------------------------------------------------------------------------
print('Reading --> ', df_pred_save_filename)
df_pred = pd.read_csv(df_pred_save_filename, index_col=0)
prediction_all_ch = df_pred.values.T.tolist()

# confusion matrix plotting --------------------------------------------------------------------------------------------
conf_mat = []

# for all channel
for ch in prediction_all_ch:
    label_count_per_ch = []
    # count for all labels
    for label in range(6):
        label_count_per_ch.append(ch.count(label))

    conf_mat.append(label_count_per_ch)

conf_mat = np.array(conf_mat).T

fig = plot_confusion_matrix(cm=conf_mat,
                            col_label=['sensor@[-3m]',
                                       'sensor@[-2m]',
                                       'sensor@[2m]',
                                       'sensor@[4m]',
                                       'sensor@[6m]',
                                       'sensor@[8m]',
                                       'sensor@[10m]'],
                            row_label=['No Leak',
                                       'Leak@[2m]',
                                       'Leak@[4.5m]',
                                       'Leak@[5m]',
                                       'Leak@[8m]',
                                       'Leak@[10m]'])

plt.show()


# classification plot along raw AE -------------------------------------------------------------------------------------
# multiple graph plot - retrieved and modified from helper.plot_multiple_timeseries()
# config
multiple_timeseries = prediction_all_ch
main_title = 'Model prediction (4) 2bar by 6k Sliding Window, Stride: {}'.format(window_stride)
subplot_titles = ['-3m', '-2m', '2m', '4m', '6m', '8m', '10m']

# do the work
time_plot_start = time.time()
no_of_plot = len(multiple_timeseries)
fig = plt.figure(figsize=(5, 8))
fig.suptitle(main_title, fontweight="bold", size=8)
fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.03)
# first plot
ax1 = fig.add_subplot(no_of_plot, 1, 1)
ax1.plot(multiple_timeseries[0])
ax1.set_title(subplot_titles[0], size=8)
ax1.set_ylim(bottom=0, top=5)
ax1.yaxis.set_ticks([0, 1, 2, 3, 4, 5])
ax1.grid('on')

# the rest of the plot
for i in range(1, no_of_plot, 1):
    ax = fig.add_subplot(no_of_plot, 1, i+1, sharex=ax1)
    ax.plot(multiple_timeseries[i])
    ax.set_title(subplot_titles[i], size=8)
    ax.set_ylim(bottom=0, top=5)
    ax.yaxis.set_ticks([0, 1, 2, 3, 4, 5])
    ax.grid('on')


plt.show()

time_plot = time.time() - time_plot_start
print('Time taken to plot: {:.4f}'.format(time_plot))

# layering misclassified position on raw ae ----------------------------------------------------------------------------
# # this is the correct label for each channels in AE data
# actual_class_per_ch = [0, 0, 0, 0, 0, 0, 0]
# label_to_dist = {0: 'nonLCP',
#                  1: '2m',
#                  2: '4.5m',
#                  3: '5m',
#                  4: '8m',
#                  5: '10m'}
#
# faulty_index_al_ch, false_class_al_ch = [], []
# # for all ch
# for pred_per_ch, actual in zip(prediction_all_ch, actual_class_per_ch):
#     temp2, temp3 = [], []
#     for index, pred in zip(window_index, pred_per_ch):
#         if pred != actual:
#             temp2.append(index)
#             temp3.append(pred)
#     faulty_index_al_ch.append(temp2)
#     false_class_al_ch.append(temp3)
#
# # print('Faulty Index all ch dim: ', np.array(faulty_index_al_ch).shape)
# # print(len(faulty_index_al_ch))
# # for j in faulty_index_al_ch:
# #     print(len(j))
#
# main_title = 'Model prediction by 6k Sliding Window, Stride: {}'.format(window_stride)
# subplot_titles = ['-3m', '-2m', '2m', '4m', '6m', '8m', '10m', '12m']
# raw_ae = n_channel_data
# # do the work
# time_plot_start = time.time()
# no_of_plot = len(faulty_index_al_ch)
# fig = plt.figure(figsize=(5, 8))
# fig.suptitle(main_title, fontweight="bold", size=8)
# fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.03)
# # first plot
# ax1 = fig.add_subplot(no_of_plot, 1, 1)
# ax1.plot(raw_ae[0], c='b')
# # plot marker for false positive
# if len(faulty_index_al_ch[0]) > 0:
#     ax1.plot(faulty_index_al_ch[0], np.zeros(len(faulty_index_al_ch[0])), c='r', marker='x', linestyle='None')
#
#     # annotate the misclassified class
#     for fc, x in zip(false_class_al_ch[0], faulty_index_al_ch[0]):
#         ax1.annotate(fc, (x, 0.5))
#
# ax1.set_title(subplot_titles[0], size=8)
# ax1.set_ylim(bottom=-1, top=1)
#
# # the rest of the plot
# for i in range(1, no_of_plot, 1):
#     ax = fig.add_subplot(no_of_plot, 1, i+1, sharex=ax1, sharey=ax1)
#     ax.plot(raw_ae[i], c='b')
#     if len(faulty_index_al_ch[i]) > 0:
#         ax.plot(faulty_index_al_ch[i], np.zeros(len(faulty_index_al_ch[i])), c='r', marker='x', linestyle='None')
#
#         # annotate the misclassified class
#         for fc, x in zip(false_class_al_ch[i], faulty_index_al_ch[i]):
#             ax.annotate(label_to_dist[fc], (x, 0.2))
#
#     ax.set_title(subplot_titles[i], size=8)
#     ax.set_ylim(bottom=-1, top=1)
#
# plt.show()
#
# time_plot = time.time() - time_plot_start
# print('Time taken to plot: {:.4f}'.format(time_plot))