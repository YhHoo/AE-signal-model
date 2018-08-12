# This is juz for testing out small function from dsp_tools before integration on bigger one

import numpy as np
from scipy.signal import chirp, cwt, ricker
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import pywt
# self lib
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.dsp_tools import spectrogram_scipy, fft_scipy, one_dim_xcor_2d_input
from src.utils.helpers import direct_to_dir, read_all_tdms_from_folder, plot_two_heatmap_in_one_column

ae_data = AcousticEmissionDataSet_13_7_2018(drive='F')
n_channel_leak = ae_data.test_data(sensor_dist='near', pressure=1, leak=True)
n_channel_noleak = ae_data.test_data(sensor_dist='near', pressure=1, leak=False)

# wavelet scale
m_wavelet = 'gaus1'
scale = np.linspace(2, 30, 50)
fs = 1e6

stft_bank = []
for i in range(n_channel_leak.shape[0]):
    _, _, leak_stft, _ = spectrogram_scipy(sampled_data=n_channel_leak[i, 0:500000],
                                           fs=1e6,
                                           nperseg=100,
                                           vis_max_freq_range=100e3,
                                           noverlap=0,
                                           nfft=500,
                                           return_plot=False,
                                           mode='magnitude',
                                           verbose=False)
    stft_bank.append(leak_stft)

fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot([0.1, 0.2, 0.8, 0.3])
ax2 = fig.add_subplot([0.1, 0.6, 0.8, 0.3])
colorbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.01])
i = ax1.imshow(map, cmap='seismic', aspect='auto')

# data slicing and reading
# data_dir = direct_to_dir(where='yh_laptop_test_data') + 'plb/'
# n_channel_data_near = read_all_tdms_from_folder(data_dir)
# n_channel_data_near = np.swapaxes(n_channel_data_near, 1, 2)
# n_channel_data_near = n_channel_data_near[0]
# print(n_channel_data_near.shape)

# CWT --> XCOR---------------
op_1 = False
if op_1:
    # data slicing and reading
    data_dir = direct_to_dir(where='yh_laptop_test_data') + 'plb/'
    n_channel_data_near = read_all_tdms_from_folder(data_dir)
    n_channel_data_near = np.swapaxes(n_channel_data_near, 1, 2)
    # wavelet scale
    m_wavelet = 'gaus1'
    scale = np.linspace(2, 30, 50)
    fs = 1e6
    sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]

    n_channel_cwt = []
    for sensor in range(n_channel_data_near.shape[1]):
        n_channel_cwt.append(pywt.cwt(n_channel_data_near[0, sensor, 90000:130000],
                                      scales=scale, wavelet=m_wavelet, sampling_period=1/fs)[0])
    n_channel_cwt = np.array(n_channel_cwt)
    print(n_channel_cwt.shape)

    # xcor
    xcor, _ = one_dim_xcor_2d_input(input_mat=n_channel_cwt, pair_list=sensor_pair_near, verbose=True)
    dist = 0
    for map in xcor:
        fig2 = plt.figure()
        title = 'XCOR_CWT_DistDiff[{}m]'.format(dist)
        fig2.suptitle(title)
        ax1 = fig2.add_axes([0.1, 0.6, 0.8, 0.1])
        ax2 = fig2.add_axes([0.1, 0.8, 0.8, 0.1])
        cwt_ax = fig2.add_axes([0.1, 0.2, 0.8, 0.3])
        colorbar_ax = fig2.add_axes([0.1, 0.1, 0.8, 0.01])
        # title
        ax1.set_title('Sensor Index: {}'.format(sensor_pair_near[dist][0]))
        ax2.set_title('Sensor Index: {}'.format(sensor_pair_near[dist][1]))
        cwt_ax.set_title('Xcor of CWT')
        # plot
        ax1.plot(n_channel_data_near[0, sensor_pair_near[dist][0], 90000:130000])
        ax2.plot(n_channel_data_near[0, sensor_pair_near[dist][1], 90000:130000])
        cwt_ax.grid(linestyle='dotted')
        cwt_ax.axvline(x=xcor.shape[2]//2 + 1, linestyle='dotted')
        cwt_ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        i = cwt_ax.imshow(map, cmap='seismic', aspect='auto')
        plt.colorbar(i, cax=colorbar_ax, orientation='horizontal')
        # saving
        filename = direct_to_dir(where='result') + title
        fig2.savefig(filename)

        plt.close('all')

        print('SAVED --> ', title)
        dist += 1


# ax1.set_title('Time Series ')
# ax2.set_title('CWT using {}'.format(m_wavelet))
# ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax1.plot(signal)
# # extent=[x_start, x_end, y_start, y_end]
# i = ax2.imshow(cwt_out, extent=[0, 40000, freq[-1], freq[0]], cmap='seismic', aspect='auto')
# plt.colorbar(i, cax=colorbar_ax, orientation='horizontal')

# _, _, _, fig2 = spectrogram_scipy(sampled_data=signal, fs=1e6, nperseg=100,
#                                   noverlap=0, nfft=500, return_plot=True, mode='magnitude', verbose=True)








