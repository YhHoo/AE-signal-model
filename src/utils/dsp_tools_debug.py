# This is juz for testing out small function from dsp_tools before integration on bigger one

import numpy as np
from scipy.signal import chirp, cwt, ricker
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import pywt
# self lib
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.dsp_tools import spectrogram_scipy, fft_scipy, one_dim_xcor_2d_input, signal_smoothing_wavelet
from src.utils.helpers import direct_to_dir, read_all_tdms_from_folder, plot_heatmap_series_in_one_column, \
                              plot_multiple_timeseries, plot_cwt_with_time_series, read_single_tdms, \
                              plot_multiple_level_decomposition
from src.controlled_dataset.ideal_dataset import sine_wave_continuous, white_noise
from statsmodels.robust import mad


def waveletSmooth(x, wavelet="db4", level=1, title=None):
    # calculate the wavelet coefficients (take level as dwt_max_level)
    coeff = pywt.wavedec(x, wavelet, mode="per")
    # calculate a threshold (Median Absolute Deviation) - find deviation of every item from median in a 1d array, then
    # find the median of the deviation again.
    sigma = mad(coeff[-level])
    # changing this threshold also changes the behavior. This is universal threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    # thresholding on all the detail components, using universal threshold from the level (-level)
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec(coeff, wavelet, mode="per")
    f, ax = plt.subplots()
    ax.plot(x, color="b", alpha=0.5)
    ax.plot(y, color="r")
    if title:
        ax.set_title(title)
    ax.set_xlim((0, len(y)))

    plt.show()


def bumps(x):
    """
    A sum of bumps with locations t at the same places as jumps in blocks.
    The heights h and widths s vary and the individual bumps are of the
    form K(t) = 1/(1+|x|)**4
    """
    K = lambda x : (1. + np.abs(x)) ** -4.
    t = np.array([[.1, .13, .15, .23, .25, .4, .44, .65, .76, .78, .81]]).T
    h = np.array([[4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 2.1, 4.2]]).T
    w = np.array([[.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005]]).T
    return np.sum(h*K((x-t)/w), axis=0)


# time domain setting
fs = 1000
time_duration = 1
time_axis = np.linspace(0, time_duration, time_duration*fs)

# bump signal
bump_sig = bumps(x=time_axis)


# noise signal
w_noise = white_noise(time_axis=time_axis, power=0.01)
# sine = sine_wave_continuous(time_axis=time_axis, amplitude=2, fo=100, phase=0)

# summing 2 signal
sig_mix = []
for i, j in zip(bump_sig, w_noise):
    sig_mix.append(i+j)

# denoised_sig = signal_smoothing_wavelet(x=sig_mix)
# print(denoised_sig.shape)


# wavelet decomposition
w = pywt.Wavelet('db4')
dec_level = 3
print('MAX DEC LEVEL: ', pywt.dwt_max_level(data_len=len(time_axis), filter_len=w.dec_len))
coeff = pywt.wavedec(sig_mix, w, mode="per", level=dec_level)
dec_titles = ['cA_{}'.format(dec_level)] + ['cD_{}'.format(i) for i in range(dec_level, 0, -1)]
print(dec_titles)
fig = plot_multiple_level_decomposition(ori_signal=sig_mix,
                                        dec_signal=coeff,
                                        dec_titles=dec_titles,
                                        main_title='wavelet decomposition')

f_mag, _, f_axis = fft_scipy(sampled_data=coeff[0],
                             fs=fs,
                             visualize=False)
fig2 = plt.figure()
ax = fig2.add_subplot(1, 1, 1)
ax.plot(f_axis, f_mag)

# for arr in coeff:
#     print(arr.shape)
#
# ax_dec_1.plot(coeff[0])
# ax_dec_2.plot(coeff[-1])


# fft_scipy(sampled_data=sig_mix, fs=fs, visualize=True)
# # waveletSmooth(x=sine_w_noise, level=4)
#
# # wavelet
# m_wavelet = 'gaus2'
# scale = np.linspace(1, 30, 100)
# cwt_out, freq = pywt.cwt(sine, scales=scale, wavelet=m_wavelet, sampling_period=1 / fs)
# fig = plot_cwt_with_time_series(time_series=sine,
#                                 no_of_time_series=1,
#                                 cwt_mat=cwt_out,
#                                 cwt_scale=scale)
# print(freq)

# plt.grid(linestyle='dotted')
plt.show()



# # CWT --> XCOR (using LAPTOP PLB test data)-----------------------------------------------------------------------------
# op_1 = False
# if op_1:
#     # data slicing and reading
#     data_dir = direct_to_dir(where='yh_laptop_test_data') + 'plb/'
#     n_channel_data_near = read_all_tdms_from_folder(data_dir)
#     n_channel_data_near = np.swapaxes(n_channel_data_near, 1, 2)
#     # wavelet scale
#     m_wavelet = 'gaus1'
#     scale = np.linspace(2, 30, 50)
#     fs = 1e6
#     sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
#
#     n_channel_cwt = []
#     for sensor in range(n_channel_data_near.shape[1]):
#         n_channel_cwt.append(pywt.cwt(n_channel_data_near[0, sensor, 90000:130000],
#                                       scales=scale, wavelet=m_wavelet, sampling_period=1/fs)[0])
#     n_channel_cwt = np.array(n_channel_cwt)
#     print(n_channel_cwt.shape)
#
#     # xcor
#     xcor, _ = one_dim_xcor_2d_input(input_mat=n_channel_cwt, pair_list=sensor_pair_near, verbose=True)
#     dist = 0
#     for map in xcor:
#         fig2 = plt.figure()
#         title = 'XCOR_CWT_DistDiff[{}m]'.format(dist)
#         fig2.suptitle(title)
#         ax1 = fig2.add_axes([0.1, 0.6, 0.8, 0.1])
#         ax2 = fig2.add_axes([0.1, 0.8, 0.8, 0.1])
#         cwt_ax = fig2.add_axes([0.1, 0.2, 0.8, 0.3])
#         colorbar_ax = fig2.add_axes([0.1, 0.1, 0.8, 0.01])
#         # title
#         ax1.set_title('Sensor Index: {}'.format(sensor_pair_near[dist][0]))
#         ax2.set_title('Sensor Index: {}'.format(sensor_pair_near[dist][1]))
#         cwt_ax.set_title('Xcor of CWT')
#         # plot
#         ax1.plot(n_channel_data_near[0, sensor_pair_near[dist][0], 90000:130000])
#         ax2.plot(n_channel_data_near[0, sensor_pair_near[dist][1], 90000:130000])
#         cwt_ax.grid(linestyle='dotted')
#         cwt_ax.axvline(x=xcor.shape[2]//2 + 1, linestyle='dotted')
#         cwt_ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#         i = cwt_ax.imshow(map, cmap='seismic', aspect='auto')
#         plt.colorbar(i, cax=colorbar_ax, orientation='horizontal')
#         # saving
#         filename = direct_to_dir(where='result') + title
#         fig2.savefig(filename)
#
#         plt.close('all')
#
#         print('SAVED --> ', title)
#         dist += 1
#
#
# # CWT --> XCOR (using LAPTOP Leak no Leak test data)--------------------------------------------------------------------
# op_2 = False
# if op_2:
#     # wavelet scale
#     m_wavelet = 'gaus1'
#     scale = np.linspace(2, 30, 50)
#     fs = 1e6
#
#     # leak data
#     data_dir = direct_to_dir(where='yh_laptop_test_data') + '1bar_leak/'
#     n_channel_leak = read_all_tdms_from_folder(data_dir)
#     n_channel_leak = np.swapaxes(n_channel_leak, 1, 2)
#     n_channel_leak = n_channel_leak[0, 1:3, :100000]
#     # no leak data
#     data_dir = direct_to_dir(where='yh_laptop_test_data') + '1bar_noleak/'
#     n_channel_noleak = read_all_tdms_from_folder(data_dir)
#     n_channel_noleak = np.swapaxes(n_channel_noleak, 1, 2)
#     n_channel_noleak = n_channel_noleak[0, 1:3, :100000]
#
#     # break into a list of segmented points
#     no_of_segment = 10
#     n_channel_leak = np.split(n_channel_leak, axis=1, indices_or_sections=no_of_segment)
#     n_channel_noleak = np.split(n_channel_noleak, axis=1, indices_or_sections=no_of_segment)
#
#     print(len(n_channel_leak))
#     cwt_bank_pos1, cwt_bank_pos2 = [], []
#     # for leak data at -2 and 2m only
#     for segment in n_channel_leak:
#         pos1_leak_cwt, _ = pywt.cwt(segment[0], scales=scale, wavelet=m_wavelet, sampling_period=1 / fs)
#         pos2_leak_cwt, _ = pywt.cwt(segment[1], scales=scale, wavelet=m_wavelet, sampling_period=1 / fs)
#         cwt_bank_pos1.append(pos1_leak_cwt)
#         cwt_bank_pos2.append(pos2_leak_cwt)
#
#     # xcor
#     cwt_xcor_bank = []
#     max_point = []
#     for cwt_pair in zip(cwt_bank_pos1, cwt_bank_pos2):
#         xcor, _ = one_dim_xcor_2d_input(input_mat=np.array([cwt_pair[0], cwt_pair[1]]), pair_list=[(0, 1)])
#         cwt_xcor_bank.append(xcor[0])
#         max_point.append(np.unravel_index(np.argmax(xcor[0], axis=None), xcor[0].shape))
#
#     # visualizing
#     for i in range(no_of_segment):
#         fig2 = plot_cwt_with_time_series(time_series=[n_channel_leak[i][0, :], n_channel_leak[i][1, :]],
#                                          no_of_time_series=2,
#                                          cwt_mat=cwt_xcor_bank[i],
#                                          cwt_scale=scale,
#                                          title='XCOR OF CWT OF 2 TIME SERIES, Sample[{}]'.format(i))
#         plt.show()
#
# # scaling of CWT output
# op_3 = False
# if op_3:
#     # wavelet scale
#     m_wavelet = 'gaus1'
#     scale = np.linspace(2, 30, 50)
#     fs = 1e6
#
#     # leak data
#     data_dir = direct_to_dir(where='yh_laptop_test_data') + '1bar_leak/'
#     n_channel_leak = read_all_tdms_from_folder(data_dir)
#     n_channel_leak = np.swapaxes(n_channel_leak, 1, 2)
#     n_channel_leak = n_channel_leak[0, 1:3, :100000]
#
#     # break into a list of segmented points
#     no_of_segment = 10
#     n_channel_leak = np.split(n_channel_leak, axis=1, indices_or_sections=no_of_segment)
#
#     pos1_leak_cwt, f_axis = pywt.cwt(n_channel_leak[0][0], scales=scale, wavelet=m_wavelet, sampling_period=1 / fs)
#     # f_axis = np.flip(f_axis, axis=0)
#
#     fig = plot_cwt_with_time_series(time_series=n_channel_leak[0][0, :], no_of_time_series=1,
#                                     cwt_mat=pos1_leak_cwt, cwt_scale=scale)
#
#     # print('CWT Dim: ', pos1_leak_cwt.shape)
#     # print('Scale: ', scale)
#     # print('freq_axis: ', f_axis)
#     #
#     # # visualize
#     # fig = plt.figure()
#     # title = 'XCOR_CWT_DistDiff[0m]'
#     # fig.suptitle(title, fontweight='bold')
#     # ax2 = fig.add_axes([0.1, 0.8, 0.8, 0.1])
#     # cwt_ax = fig.add_axes([0.1, 0.2, 0.8, 0.5], sharex=ax2)
#     # colorbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.01])
#     # # title
#     # ax2.set_title('Sensor[-2m]')
#     # cwt_ax.set_title('Xcor of CWT')
#     # # plot
#     # ax2.plot(n_channel_leak[0][0, :])
#     # cwt_ax.grid(linestyle='dotted')
#     # # cwt_ax.axvline(x=pos1_leak_cwt.shape[1] // 2 + 1, linestyle='dotted')
#     # # cwt_ax.scatter(max_point[i][1], max_point[i][0], s=70, c='black', marker='x')
#     # # cwt_ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#     # # cwt_ax.set_yscale('log')
#     # # cwt_ax.set_yticks(f_axis)
#     # ix = cwt_ax.imshow(pos1_leak_cwt, cmap='seismic', aspect='auto',
#     #                    extent=[0, pos1_leak_cwt.shape[1], f_axis[-1], f_axis[0]])
#     # plt.colorbar(ix, cax=colorbar_ax, orientation='horizontal')
#
#     plt.show()
#     # print(max_point[i])
#
