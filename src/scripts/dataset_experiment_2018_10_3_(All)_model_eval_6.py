'''
THIS SCRIPT IS TO FEED THE LCP RECOGNITION MODEL WITH RAW AE DATA, AND SEE WHETHER IT WILL RECORGNIZE IT WELL. MODEL
WILL SLIDE THROUGH ONE 5M POINT AE RAW DATA, USING A WINDOW, WITH A STRIDE.
DATASET: BINARY CLASSIFICATION MODEL LNL (UNSEEN leak)
'''

import gc
import time
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

# saving naming
model_name = 'LNL_5x1'  # *
lcp_model = load_model(model_name=model_name)
lcp_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
print(lcp_model.summary())

# file reading
all_tdms_dir = 'F:/Experiment_21_12_2018/8Ch/-3,-2,0,5,7,15,16/2 bar/Leak/Test data/'
all_tdms = [(all_tdms_dir + f) for f in listdir(all_tdms_dir) if f.endswith('.tdms')]

# UPDATE PARAM HERE ***************************
actual_label = [1, 1, 1, 1, 1, 1, 1]  # label we expect model to produce (multiple label is acceptable)
model_possible_label = [0, 1]
# the physical meaning of the model label
model_label_to_dist = {0: 'NoLeak',
                       1: 'Leak'}
input_data_labels = ['sensor@[-3m]',  # the channels' dist of the input data **
                     'sensor@[-2m]',
                     'sensor@[0m]',
                     'sensor@[5m]',
                     'sensor@[7m]',
                     'sensor@[15m]',
                     'sensor@[16m]']
fig_cm_title = 'confusion mat (6-LEAK Test Data)'
# ***************************************

# ------------------------------------------------------------------------------------------------------------ DATA PREP
for file_to_test in all_tdms:
    x = file_to_test.split(sep='/')[-1]
    # discard the .tdms
    x = x.split(sep='.')[-2]

    filename_to_save = 'pred_result_[{}]_[{}]_L'.format(model_name, x)  # **

    # SAVING CONFIG
    df_pred_save_filename = direct_to_dir(where='result') + filename_to_save + '.csv'

    # test for near
    n_channel_data = read_single_tdms(file_to_test)
    n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1]  # drop useless channel 8
    # n_channel_data = np.delete(n_channel_data, 3, axis=0)  # drop broken channel 4m (for NoLeak ONLY)

    print('TDMS data dim: ', n_channel_data.shape)
    total_len = n_channel_data.shape[1]
    total_ch = len(n_channel_data)

    # ensure enough length of data input
    assert total_len > (window_size[0] + window_size[1]), 'Data length is too short, mz be at least {}'.\
        format(window_size[0] + window_size[1])
    window_index = np.arange(window_size[0], (total_len - window_size[1]), window_stride)
    print('Window Index Len: ', len(window_index))
    print('Window Index: ', window_index)

    # ---------------------------------------------------------------------------------------------------- EXECUTE MODEL
    # creating index of sliding windows [index-1000, index+5000]
    scaler = MinMaxScaler(feature_range=(-1, 1))

    prediction_all_ch = []
    # for all channels
    for ch_no in range(total_ch):
        temp, model_pred = [], []
        pb = ProgressBarForLoop(title='Iterating all Samples in ch[{}]'.format(ch_no), end=len(window_index))
        progress = 0
        for index in window_index:
            pb.update(now=progress)
            data = n_channel_data[ch_no, (index - window_size[0]):(index + window_size[1])]
            # data_norm = scaler.fit_transform(data.reshape(-1, 1)).ravel()  # **normalize
            temp.append(data)

            # detect for last entry
            if progress < (len(window_index) - 1):

                if len(temp) < sample_size_for_prediction:
                    progress += 1
                    continue
                else:
                    progress += 1
                    # print('temp full !')

            # do tis when temp is full
            # reshape
            temp = np.array(temp)
            temp = temp.reshape((temp.shape[0], temp.shape[1], 1))
            # print(temp.shape)
            time_predict_start = time.time()
            prediction = np.argmax(lcp_model.predict(temp), axis=1)

            # # estimation of dist using all class posterior probability
            # estimation = []
            # for p in lcp_model.predict(temp):
            #     estimation.append(p[0]*0 + p[1]*2 + p[2]*4.5 + p[3]*5 + p[4]*8 + p[5]*10)

            time_predict = time.time() - time_predict_start
            model_pred.append(prediction)
            # reset temp
            temp = []
            # free up memory
            gc.collect()

        pb.destroy()
        model_pred = np.concatenate(model_pred, axis=0)
        print('Model Prediction Dim: ', model_pred.shape)

        prediction_all_ch.append(model_pred)

    prediction_all_ch = np.array(prediction_all_ch).T
    df_pred = pd.DataFrame(data=prediction_all_ch,
                           columns=input_data_labels)
    df_pred.to_csv(df_pred_save_filename)
    print('Saved --> ', df_pred_save_filename)
    print('Reading --> ', df_pred_save_filename)
    df_pred = pd.read_csv(df_pred_save_filename, index_col=0)
    prediction_all_ch = df_pred.values.T.tolist()

    # -------------------------------------------------------------------------------------------------- CONFUSION MARIX
    conf_mat = []
    acc_per_ch = []

    # for all channel
    for ch, actual in zip(prediction_all_ch, actual_label):
        acc = 0
        label_count_per_ch = []
        # count for all labels
        for label in model_possible_label:
            count = ch.count(label)
            label_count_per_ch.append(count)
            # for actual class
            if isinstance(actual, tuple):
                if label in actual:
                    acc += count
            else:
                if label is actual:
                    acc += count
        # calc each channel classification acc
        acc_per_ch.append(acc / len(ch))
        # record the class count
        conf_mat.append(label_count_per_ch)

    conf_mat = np.array(conf_mat).T

    # merge col label with class accuracy
    col_label_w_acc = []
    for i, j in zip(input_data_labels, acc_per_ch):
        col_label_w_acc.append(i + '\nacc: {:.4f}'.format(j))

    fig_cm = plot_confusion_matrix(cm=conf_mat,
                                   col_label=col_label_w_acc,
                                   row_label=['No Leak',
                                              'Leak'],
                                   title=fig_cm_title)  # **
    fig_cm_save_filename = direct_to_dir(where='result') + 'cm_' + filename_to_save + '.png'
    fig_cm.savefig(fig_cm_save_filename)

    plt.close('all')
    print('Confusion Mat. fig saved -->', fig_cm_save_filename)
    print('\n------------------------------------------------------------\n')

# # ----------------------------------------------------------------------------- PLOT CLASSIFICATION RESULT IN SEQUENCE
# # multiple graph plot - retrieved and modified from helper.plot_multiple_timeseries()
# # config
# multiple_timeseries = prediction_all_ch
# main_title = 'Model prediction (4) 2bar by 6k Sliding Window, Stride: {}'.format(window_stride)
# subplot_titles = ['-3m', '-2m', '2m', '4m', '6m', '8m', '10m']
#
# # do the work
# time_plot_start = time.time()
# no_of_plot = len(multiple_timeseries)
# fig = plt.figure(figsize=(5, 8))
# fig.suptitle(main_title, fontweight="bold", size=8)
# fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.03)
# # first plot
# ax1 = fig.add_subplot(no_of_plot, 1, 1)
# ax1.plot(multiple_timeseries[0])
# ax1.set_title(subplot_titles[0], size=8)
# ax1.set_ylim(bottom=0, top=5)
# ax1.yaxis.set_ticks([0, 1, 2, 3, 4, 5])
# ax1.grid('on')
#
# # the rest of the plot
# for i in range(1, no_of_plot, 1):
#     ax = fig.add_subplot(no_of_plot, 1, i+1, sharex=ax1)
#     ax.plot(multiple_timeseries[i])
#     ax.set_title(subplot_titles[i], size=8)
#     ax.set_ylim(bottom=0, top=5)
#     ax.yaxis.set_ticks([0, 1, 2, 3, 4, 5])
#     ax.grid('on')
#
#
# plt.show()
#
# time_plot = time.time() - time_plot_start
# print('Time taken to plot: {:.4f}'.format(time_plot))

# --------------------------------------------------------------------------------- PLOT MISCLASSIFIED POINT + AE SIGNAL
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