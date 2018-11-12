'''
THIS SCRIPT IS TO FEED THE LCP RECOGNITION MODEL WITH RAW AE DATA, AND SEE WHETHER IT WILL RECORGNIZE IT WELL
'''
import time
import tensorflow as tf
from src.utils.helpers import *

# instruct GPU to allocate only sufficient memory for this script
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# file reading
tdms_leak_filename = 'F:/Experiment_3_10_2018/-4.5, -2, 2, 5, 8, 10, 17 (leak 1bar)/12709_test_0010.tdms'
tdms_noleak_filename = 'F:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/test1.tdms'

lcp_model = load_model(model_name='model in evaluation/LCP_Recog_1')
lcp_model.compile(loss='binary_crossentropy', optimizer='rmsprop')

print(lcp_model.summary())

n_channel_data = read_single_tdms(tdms_noleak_filename)
n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1]

print(n_channel_data.shape)

head_index = 2222659  # 2637500
data_to_detect = n_channel_data[:, head_index-1000:head_index+5000]

scaler = MinMaxScaler(feature_range=(-1, 1))
temp = []
for ch in data_to_detect:
    temp.append(scaler.fit_transform(ch.reshape(-1, 1)).ravel())

data_to_detect = np.array(temp)
print('AFTER NORMALIZE: ', data_to_detect.shape)

fig_n_channel = plot_multiple_timeseries(input=data_to_detect,
                                         subplot_titles=['-4.5m [0]', '-2m [1]', '2m [2]', '5m [3]',
                                                         '8m [4]', '10m [5]', '17m[6]'],
                                         main_title='Leak section')

fig_n_channel_2 = plot_multiple_timeseries(input=n_channel_data,
                                           subplot_titles=['-4.5m [0]', '-2m [1]', '2m [2]', '5m [3]',
                                                          '8m [4]', '10m [5]', '17m[6]'],
                                           main_title='entire signal')


data_to_detect = data_to_detect[0].reshape((1, 6000, 1))
print('AFTER RESHAPED: ', data_to_detect.shape)

detection_score = lcp_model.predict(data_to_detect)
print('MODEL PREDICTION: ', detection_score)
#
plt.show()


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
