# This is juz for testing out small function from dsp_tools before integration on bigger one

import numpy as np
from scipy.signal import chirp, cwt, ricker
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import pywt
# self lib
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.dsp_tools import spectrogram_scipy, fft_scipy, one_dim_xcor_2d_input, dwt_smoothing
from src.utils.helpers import direct_to_dir, read_all_tdms_from_folder, plot_heatmap_series_in_one_column, \
                              plot_multiple_timeseries, plot_cwt_with_time_series, read_single_tdms, \
                              plot_multiple_level_decomposition
from src.controlled_dataset.ideal_dataset import sine_wave_continuous, white_noise
from statsmodels.robust import mad


# # CWT --> XCOR (using LAPTOP PLB test data)---------------------------------------------------------------------------
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
