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

# SLIDING WINDOW CONFIG
window_stride = 10
window_size = (1000, 5000)
sample_size_for_prediction = 10000

# file reading
tdms_leak_filename = 'F:/Experiment_3_10_2018/-4.5, -2, 2, 5, 8, 10, 17 (leak 1bar)/12709_test_0010.tdms'
tdms_noleak_filename = 'F:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/test1.tdms'

lcp_model = load_model(model_name='LCP_Dist_Recog_1')
lcp_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print(lcp_model.summary())

n_channel_data = read_single_tdms(tdms_leak_filename)
n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-2]

print('TDMS data dim: ', n_channel_data.shape)

# creating index of sliding windows [index-1000, index+5000]
scaler = MinMaxScaler(feature_range=(-1, 1))
total_len = n_channel_data.shape[1]

# ensure enough length of data input
assert total_len > (window_size[0] + window_size[1]), 'Data length is too short, mz be at least {}'.\
    format(window_size[0] + window_size[1])
window_index = np.arange(window_size[0], (total_len - window_size[1]), window_stride)
print('Window Index Len: ', len(window_index))
print('Window Index: ', window_index)

prediction_all_ch = []
# for all channels
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


# multiple graph plot - retrieved and modified from helper.plot_multiple_timeseries() ----------------------------------
# config
multiple_timeseries = prediction_all_ch
main_title = 'Model prediction by 6k Sliding Window, Stride: {}'.format(window_stride)
subplot_titles = ['-4.5m', '-2m', '2m', '5m', '8m', '10m']

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

# the rest of the plot
for i in range(1, no_of_plot, 1):
    ax = fig.add_subplot(no_of_plot, 1, i+1, sharex=ax1)
    ax.plot(multiple_timeseries[i])
    ax.set_title(subplot_titles[i], size=8)
    ax.set_ylim(bottom=0, top=5)

plt.show()

time_plot = time.time() - time_plot_start
print('Time taken to plot: {:.4f}'.format(time_plot))
