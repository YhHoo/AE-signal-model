'''
THIS SCRIPT IS TO FEED THE LCP RECOGNITION MODEL WITH RAW AE DATA, AND SEE WHETHER IT WILL RECORGNIZE IT WELL
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

# file reading
tdms_leak_filename = 'F:/Experiment_3_10_2018/-4.5, -2, 2, 5, 8, 10, 17 (leak 1bar)/12709_test_0010.tdms'
tdms_noleak_filename = 'F:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/test1.tdms'

lcp_model = load_model(model_name='LCP_Dist_Recog_1')
lcp_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print(lcp_model.summary())

n_channel_data = read_single_tdms(tdms_leak_filename)
n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-2, :1000000]  # truncate to 1M point, due to long processing time

print('TDMS data dim: ', n_channel_data.shape)

# creating index of sliding windows [index-1000, index+5000]
scaler = MinMaxScaler(feature_range=(-1, 1))
window_stride = 10
window_size = (1000, 5000)
sample_size_for_prediction = 10000
total_len = n_channel_data.shape[1]

# ensure enough length of data input
assert total_len > (window_size[0] + window_size[1]), 'Data length is too short, mz be at least {}'.\
    format(window_size[0] + window_size[1])
window_index = np.arange(window_size[0], (total_len - window_size[1]), window_stride)
print('Window Index Len: ', len(window_index))
print('Window Index: ', window_index)

prediction_all_ch = []
for ch_no in range(6):
    temp, model_pred = [], []
    pb = ProgressBarForLoop(title='Iterating all Samples in ch[{}]'.format(ch_no), end=len(window_index))
    progress = 0
    for index in window_index:
        pb.update(now=progress)
        data = n_channel_data[ch_no, (index - window_size[0]):(index + window_size[1])]
        data_norm = scaler.fit_transform(data.reshape(-1, 1)).ravel()
        temp.append(data_norm)

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
        time_predict = time.time() - time_predict_start
        # print('Time taken for Predicting {} samples: {:.4f}s'.format(sample_size_for_prediction, time_predict))
        # print(prediction.shape)
        model_pred.append(prediction)
        # reset temp
        temp = []
        # free up memory
        gc.collect()

    pb.destroy()
    model_pred = np.concatenate(model_pred, axis=0)
    print('Model Prediction Dim: ', model_pred.shape)

    prediction_all_ch.append(model_pred)

color_seq = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
color_label = ['-4.5m', '-2m', '2m', '5m', '8m', '10m']
color_dict = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y'}
r_leg = mpatches.Patch(color='r', label=color_label[0])
g_leg = mpatches.Patch(color='g', label=color_label[1])
b_leg = mpatches.Patch(color='b', label=color_label[2])
c_leg = mpatches.Patch(color='c', label=color_label[3])
m_leg = mpatches.Patch(color='m', label=color_label[4])
y_leg = mpatches.Patch(color='y', label=color_label[5])

figure_result = plt.figure(figsize=(5, 8))
figure_result.suptitle('Model prediction by 6k Sliding Window, Stride: {}'.format(window_stride),
                       fontweight="bold",
                       size=8)
figure_result.subplots_adjust(hspace=0.7, top=0.9, bottom=0.03)
ax1 = figure_result.add_subplot(6, 1, 1)
ax1.set_title(color_label[0])
ax1.plot(n_channel_data[0])

# label the result
for pred_index, pred in zip(window_index, prediction_all_ch[0]):
    ax1.axvline(pred_index, color=color_dict[pred])

for plot_no, plot_label, raw_signal, prediction_per_ch in zip(np.arange(2, 7, 1),
                                                              color_label[1:],
                                                              n_channel_data[1:],
                                                              prediction_all_ch[1:]):
    ax2 = figure_result.add_subplot(6, 1, plot_no, sharex=ax1)
    ax2.set_title(plot_label)
    ax2.plot(raw_signal)

    # label the result
    for pred_index, pred in zip(window_index, prediction_per_ch):
        ax1.axvline(pred_index, color=color_dict[pred])


plt.legend(handles=[r_leg, g_leg, b_leg, c_leg, m_leg, y_leg])
plt.show()


# head_index = 2222659  # 2637500
# data_to_detect = n_channel_data[:, head_index-1000:head_index+5000]
#
# scaler = MinMaxScaler(feature_range=(-1, 1))
# temp = []
# for ch in data_to_detect:
#     temp.append(scaler.fit_transform(ch.reshape(-1, 1)).ravel())
#
# data_to_detect = np.array(temp)
# print('AFTER NORMALIZE: ', data_to_detect.shape)
#
# fig_n_channel = plot_multiple_timeseries(input=data_to_detect,
#                                          subplot_titles=['-4.5m [0]', '-2m [1]', '2m [2]', '5m [3]',
#                                                          '8m [4]', '10m [5]'],
#                                          main_title='Leak section')
#
# fig_n_channel_2 = plot_multiple_timeseries(input=n_channel_data,
#                                            subplot_titles=['-4.5m [0]', '-2m [1]', '2m [2]', '5m [3]',
#                                                            '8m [4]', '10m [5]'],
#                                            main_title='entire signal')
#
#
# data_to_detect = data_to_detect[0].reshape((1, 6000, 1))
# print('AFTER RESHAPED: ', data_to_detect.shape)
#
# detection_score = lcp_model.predict(data_to_detect)
# print('MODEL PREDICTION: ', detection_score)
# #
# plt.show()


# data_in_window = slide_window(seq=n_channel_data[0, :], n=6000)
#
# peak_detected = []
# window_peak_index = 1000
# time_detect_s = time.time()
# for i in data_in_window:
#     data_to_detect = n_channel_data[0, :6000].reshape((1, 6000, 1))
#     detection_score = lcp_model.predict(data_to_detect)
#
#     if detection_score > 0.5:
#         peak_detected.append(window_peak_index[0])
#
#     window_peak_index += 1
#
# print(len(peak_detected))
#
# # duplicate the list into all ch
# detection_all_ch = []
# for i in range(len(n_channel_data)):
#     detection_all_ch.append(peak_detected)
#
# time_detect = time.time() - time_detect_s
# print('Detection time taken : {:.4f}'.format(time_detect))
#
# fig = plot_multiple_timeseries_with_roi(input=n_channel_data,
#                                         subplot_titles=['-4.5m [0]', '-2m [1]', '2m [2]', '5m [3]',
#                                                         '8m [4]', '10m [5]', '17m[6]'],
#                                         main_title='Leak in RAW lcp_model detection',
#                                         all_ch_peak=detection_all_ch)
#
# plt.show()

