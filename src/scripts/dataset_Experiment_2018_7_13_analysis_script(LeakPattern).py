import matplotlib.pyplot as plt
import pywt
import numpy as np
# self lib
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.helpers import direct_to_dir, read_all_tdms_from_folder, plot_cwt_with_time_series
from src.utils.dsp_tools import one_dim_xcor_2d_input


# CONFIG -------------
# wavelet
m_wavelet = 'gaus1'
scale = np.linspace(2, 30, 50)
fs = 1e6

# segmentation
no_of_segment = 50

# DATA POINT ---------
# leak data
# data = AcousticEmissionDataSet_13_7_2018(drive='F')
# n_channel_leak = data.test_data(sensor_dist='near', pressure=1, leak=True)

data_dir = direct_to_dir(where='yh_laptop_test_data') + '1bar_leak/'
n_channel_leak = read_all_tdms_from_folder(data_dir)
n_channel_leak = np.swapaxes(n_channel_leak, 1, 2)
n_channel_leak = n_channel_leak[0, 1:3, :]

# break into a list of segmented points
n_channel_leak = np.split(n_channel_leak, axis=1, indices_or_sections=no_of_segment)
print('Total Segment: ', len(n_channel_leak))

# CWT + XCOR + VISUALIZE SCRIPT ---------------
cwt_bank_pos1, cwt_bank_pos2 = [], []
for segment in n_channel_leak:
    pos1_leak_cwt, _ = pywt.cwt(segment[0], scales=scale, wavelet=m_wavelet, sampling_period=1 / fs)
    pos2_leak_cwt, _ = pywt.cwt(segment[1], scales=scale, wavelet=m_wavelet, sampling_period=1 / fs)
    cwt_bank_pos1.append(pos1_leak_cwt)
    cwt_bank_pos2.append(pos2_leak_cwt)

# xcor
cwt_xcor_bank = []
max_point = []
for cwt_pair in zip(cwt_bank_pos1, cwt_bank_pos2):
    xcor, _ = one_dim_xcor_2d_input(input_mat=np.array([cwt_pair[0], cwt_pair[1]]), pair_list=[(0, 1)])
    cwt_xcor_bank.append(xcor[0])
    max_point.append(np.unravel_index(np.argmax(xcor[0], axis=None), xcor[0].shape))

# visualizing
for i in range(no_of_segment):
    fig2 = plot_cwt_with_time_series(time_series=[n_channel_leak[i][0, :], n_channel_leak[i][1, :]],
                                     no_of_time_series=2,
                                     cwt_mat=cwt_xcor_bank[i],
                                     cwt_scale=scale,
                                     title='XCOR OF CWT OF 2 TIME SERIES, Sample[{}]'.format(i))
    filename = direct_to_dir(where='google_drive') + 'xcor_cwt_sample[{}]'.format(i)
    fig2.savefig(filename)
    plt.close('all')
