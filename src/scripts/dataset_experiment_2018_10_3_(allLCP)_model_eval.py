'''
THIS SCRIPT IS TO FEED THE LCP RECOGNITION MODEL WITH RAW AE DATA, AND SEE WHETHER IT WILL RECORGNIZE IT WELL
'''

from src.utils.helpers import *


# file reading
tdms_leak_filename = 'F:/Experiment_3_10_2018/-4.5, -2, 2, 5, 8, 10, 17 (leak 1bar)/12709_test_0010.tdms'
tdms_noleak_filename = 'F:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/test1.tdms'

lcp_model = load_model(model_name='model in evaluation/LCP_Recog_1')
lcp_model.compile(loss='binary_crossentropy', optimizer='rmsprop')

print(lcp_model.summary())


n_channel_data = read_single_tdms(tdms_leak_filename)
n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1]

print(n_channel_data.shape)

# fig_n_channel = plot_multiple_timeseries(input=n_channel_data,
#                                          subplot_titles=['-4.5m [0]', '-2m [1]', '2m [2]', '5m [3]',
#                                                          '8m [4]', '10m [5]', '17m[6]'],
#                                          main_title='Leak in RAW')
#
# plt.show()


data_in_window = slide_window(seq=n_channel_data[0, :], n=6000)

for i in data_in_window:
