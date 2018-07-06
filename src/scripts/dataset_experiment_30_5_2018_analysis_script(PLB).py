'''
This script is used to analyze the PLB data sets from experiment 30_5_2018. We aim to finds the recognizable 2d patterns
from the plb time series data, either by STFT in magnitude, phase... and Wavelet.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, ricker, find_peaks_cwt
# self lib
from src.experiment_dataset.dataset_experiment_30_5_2018 import AcousticEmissionDataSet_30_5_2018
from src.utils.dsp_tools import one_dim_xcor_2d_input, butter_bandpass_filtfilt
from src.utils.helpers import three_dim_visualizer
from src.utils.plb_analysis_tools import dual_sensor_xcor_with_stft_qiuckview


# -------------------[PLB TEST]-------------------
data = AcousticEmissionDataSet_30_5_2018(drive='F')
# set_no = [1, 1, 2, 1]
segment = [(1080000, 870000, 660000),
           (700000, 920000, 720000),
           (370000, 1080000, 1000000),
           (700000, 1130000, 880000)]  # +100e3
pos = [0, 2, 4, 6]
widths_2 = np.arange(1, 20, 1)
savepath = 'C:/Users/YH/Desktop/hooyuheng.masterWork/MASTER_PAPERWORK/' \
           'My Practical Work------------/Exp30_5_2018/PLB test/Real Data/STFT + Xcor/temp/'

# TEMP DEBUGGING THE SENSOR DATA CWT @ 4M <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
leak_pos = 0
n_channel_data, _, _, _ = data.plb_4_sensor(leak_pos=leak_pos)
start = segment[0][0]
input_signal_1 = n_channel_data[0, start:start+100000, 1]
input_signal_2 = n_channel_data[0, start:start+100000, 2]
start = segment[0][1]
input_signal_3 = n_channel_data[1, start:start+100000, 1]
input_signal_4 = n_channel_data[1, start:start+100000, 2]
start = segment[0][2]
input_signal_5 = n_channel_data[2, start:start+100000, 1]
input_signal_6 = n_channel_data[2, start:start+100000, 2]
# bandpass from 20kHz to 100kHz
filtered_signal_1 = butter_bandpass_filtfilt(sampled_data=input_signal_1, fs=1e6, f_hicut=1e5, f_locut=20e3)
filtered_signal_2 = butter_bandpass_filtfilt(sampled_data=input_signal_2, fs=1e6, f_hicut=1e5, f_locut=20e3)
filtered_signal_3 = butter_bandpass_filtfilt(sampled_data=input_signal_3, fs=1e6, f_hicut=1e5, f_locut=20e3)
filtered_signal_4 = butter_bandpass_filtfilt(sampled_data=input_signal_4, fs=1e6, f_hicut=1e5, f_locut=20e3)
filtered_signal_5 = butter_bandpass_filtfilt(sampled_data=input_signal_5, fs=1e6, f_hicut=1e5, f_locut=20e3)
filtered_signal_6 = butter_bandpass_filtfilt(sampled_data=input_signal_6, fs=1e6, f_hicut=1e5, f_locut=20e3)

signal_input_pair = [(filtered_signal_1, filtered_signal_2),
                     (filtered_signal_3, filtered_signal_4),
                     (filtered_signal_5, filtered_signal_6)]
set_no = 0
for signal in signal_input_pair:
    fig_time, fig_stft_1, fig_stft_2, fig_xcor = dual_sensor_xcor_with_stft_qiuckview(data_1=signal[0],
                                                                                      data_2=signal[1],
                                                                                      stft_mode='magnitude',
                                                                                      stft_nperseg=100,
                                                                                      plot_label=['0m', '-1m', '22m'])
    path_temp = '{}Sensor[-1m]_leak[{}m]_set{}'.format(savepath, leak_pos, set_no)
    fig_stft_1.savefig(path_temp)
    path_temp = '{}Sensor[22m]_leak[{}m]_set{}'.format(savepath, leak_pos, set_no)
    fig_stft_2.savefig(path_temp)
    path_temp = '{}XcorMap_leak[{}m]_set{}'.format(savepath, leak_pos, set_no)
    fig_xcor.savefig(path_temp)
    set_no += 1
    print('saved')
    plt.close('all')

# fig_cwt_1 = three_dim_visualizer(x_axis=np.arange(1, cwtmatr_1.shape[1] + 1, 1),
#                                  y_axis=widths_2,
#                                  zxx=cwtmatr_1,
#                                  label=['time steps', 'Wavelet Width', 'CWT Coefficient'],
#                                  output='2d',
#                                  title='CWT Coef of Sensor[-1m], Source @ {}m'.format(pos[2]))
# fig_cwt_2 = three_dim_visualizer(x_axis=np.arange(1, cwtmatr_2.shape[1] + 1, 1),
#                                  y_axis=widths_2,
#                                  zxx=cwtmatr_2,
#                                  label=['time steps', 'Wavelet Width', 'CWT Coefficient'],
#                                  output='2d',
#                                  title='CWT Coef of Sensor[22m], Source @ {}m'.format(pos[2]))
# path_s1 = '{}DEBUG 1'.format(savepath)
# path_s2 = '{}DEBUG 2'.format(savepath)
# path_s3 = '{}DEBUG 3'.format(savepath)
# fig_cwt_1.savefig(path_s1)
# fig_cwt_2.savefig(path_s2)
# fig_time_series.savefig(path_s3)
# plt.close()
# print('Saved !')
# DEBUG END HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# -------------------[Xcor With STFT]-------------------
stft_analysis = False
if stft_analysis:
    # for all leak pos
    for p in pos:
        # loading data fr drive
        n_channel_data, _, _, _ = data.plb_4_sensor(leak_pos=p)
        # index the segment
        seg = 0
        # for all 3 sets
        for i in range(3):
            start = segment[seg][i]
            # segmenting the signal for 100k points
            input_signal_1 = n_channel_data[i, start:start + 100000, 1]
            input_signal_2 = n_channel_data[i, start:start + 100000, 2]
            # bandpass the signal
            filtered_signal_1 = butter_bandpass_filtfilt(sampled_data=input_signal_1, fs=1e6, f_hicut=1e5, f_locut=20e3)
            filtered_signal_2 = butter_bandpass_filtfilt(sampled_data=input_signal_2, fs=1e6, f_hicut=1e5, f_locut=20e3)
            # stft + xcor
            _, _, _, fig4 = dual_sensor_xcor_with_stft_qiuckview(data_1=filtered_signal_1,
                                                                 data_2=filtered_signal_2,
                                                                 stft_mode='magnitude',
                                                                 stft_nperseg=100,
                                                                 plot_label=['{}m'.format(p), '-1m', '22m'],
                                                                 save_selection=[0, 0, 0, 0])
            path_temp = '{}XcorMap_leak[{}m]_set{}'.format(savepath, p, i)
            fig4.savefig(path_temp)
            plt.close('all')
            print('XcorMap_leak[{}m]_set{} --> saved'.format(p, i))
        seg += 1

# TEMP DEBUGGING THE SENSOR DATA STFT @ 4M <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

debug = False
if debug:
    n_channel_data, _, _, _ = data.plb_4_sensor(leak_pos=4)
    set = 1
    start = segment[2][set]
    input_signal_1 = n_channel_data[0, start:start + 100000, 1]
    input_signal_2 = n_channel_data[0, start:start + 100000, 2]
    # bandpass the signal
    filtered_signal_1 = butter_bandpass_filtfilt(sampled_data=input_signal_1, fs=1e6, f_hicut=1e5, f_locut=20e3)
    filtered_signal_2 = butter_bandpass_filtfilt(sampled_data=input_signal_2, fs=1e6, f_hicut=1e5, f_locut=20e3)
    # stft + xcor
    fig1, fig2, fig3, fig4 = dual_sensor_xcor_with_stft_qiuckview(data_1=filtered_signal_1,
                                                                  data_2=filtered_signal_2,
                                                                  stft_mode='magnitude',
                                                                  stft_nperseg=100,
                                                                  plot_label=['4m', '-1m', '22m'],
                                                                  save_selection=[0, 0, 0, 0])
    plt.show()
# -------------------[Wavelet Transform]-------------------
# widths = np.array([1, 5, 10, 15])
# widths_2 = np.arange(1, 20, 0.5)
#
# savepath = 'C:/Users/YH/Desktop/hooyuheng.masterWork/MASTER_PAPERWORK/' \
#            'My Practical Work------------/Exp30_5_2018/PLB test/'
#
# for s, p in zip(set_no, pos):
#     n_channel_data, _, _, _ = data.plb_4_sensor(leak_pos=p)
#
#     # bandpass from 20kHz to 100kHz
#     input_signal_1 = n_channel_data[s, 800000:1200000, 1]
#     input_signal_2 = n_channel_data[s, 800000:1200000, 2]
#     filtered_signal_1 = butter_bandpass_filtfilt(sampled_data=input_signal_1, fs=1e6, f_hicut=100e3, f_locut=20e3)
#     filtered_signal_2 = butter_bandpass_filtfilt(sampled_data=input_signal_2, fs=1e6, f_hicut=100e3, f_locut=20e3)
#     cwtmatr_1 = cwt(filtered_signal_1, ricker, widths_2)
#     cwtmatr_2 = cwt(filtered_signal_2, ricker, widths_2)
#     print('CWT output 1 dim: ', cwtmatr_1.shape)
#     print('CWT output 2 dim: ', cwtmatr_2.shape)
#
#     fig_cwt_1 = three_dim_visualizer(x_axis=np.arange(1, cwtmatr_1.shape[1] + 1, 1),
#                                      y_axis=widths_2,
#                                      zxx=cwtmatr_1,
#                                      label=['time steps', 'Wavelet Width', 'CWT Coefficient'],
#                                      output='2d',
#                                      title='CWT Coef of Sensor[-1m], Source @ {}m'.format(p))
#     fig_cwt_2 = three_dim_visualizer(x_axis=np.arange(1, cwtmatr_2.shape[1] + 1, 1),
#                                      y_axis=widths_2,
#                                      zxx=cwtmatr_2,
#                                      label=['time steps', 'Wavelet Width', 'CWT Coefficient'],
#                                      output='2d',
#                                      title='CWT Coef of Sensor[22m], Source @ {}m'.format(p))
#     path_s1 = '{}CWT_sensor[{}]_Source@{}m'.format(savepath, '-1m', p)
#     path_s2 = '{}CWT_sensor[{}]_Source@{}m'.format(savepath, '22m', p)
#     fig_cwt_1.savefig(path_s1)
#     fig_cwt_2.savefig(path_s2)
#     plt.close()
#     print('Saved !')
#
#     cwt_amplitude_plot = False
#     if cwt_amplitude_plot:
#         cwtmatr_1_max = np.array([i.max() for i in cwtmatr_1])
#         cwtmatr_2_max = np.array([i.max() for i in cwtmatr_2])
#         fig_cwt_amplitude = plt.figure(figsize=(8, 5))
#         ax_cwt_amp_sensor1 = fig_cwt_amplitude.add_subplot(2, 1, 1)
#         ax_cwt_amp_sensor2 = fig_cwt_amplitude.add_subplot(2, 1, 2)
#         # set title of subplots
#         ax_cwt_amp_sensor1.set_title('Maximum Coefficient in CWT of Sensor 1 data against the Widths')
#         ax_cwt_amp_sensor2.set_title('Maximum Coefficient in CWT of Sensor 2 data against the Widths')
#         # plot
#         ax_cwt_amp_sensor1.plot(widths_2, cwtmatr_1_max)
#         ax_cwt_amp_sensor2.plot(widths_2, cwtmatr_2_max)
#
#     xcor_in_cwt = True
#     if xcor_in_cwt:
#         cwt_map = np.array([cwtmatr_1, cwtmatr_2])
#         sensor_pair = [(0, 1)]
#         xcor_map = one_dim_xcor_2d_input(input_mat=cwt_map,
#                                          pair_list=sensor_pair,
#                                          verbose=True)
#         xcor_map = xcor_map[0, :, 0:xcor_map.shape[2]:10]
#         print('DEBUGGING--------->', xcor_map.shape)
#         # plotting 4 CWT channels only in a time series plot
#         plot_cwt_indi = False
#         if plot_cwt_indi:
#             # plotting components for CWT xcor result
#             fig_cwt_component = plt.figure(figsize=(8, 5))
#             fig_cwt_component.suptitle('CWT Components')
#             ax_component_1 = fig_cwt_component.add_subplot(4, 1, 1)
#             ax_component_2 = fig_cwt_component.add_subplot(4, 1, 2)
#             ax_component_3 = fig_cwt_component.add_subplot(4, 1, 3)
#             ax_component_4 = fig_cwt_component.add_subplot(4, 1, 4)
#             # ax_component_5 = fig_cwt_component.add_subplot(5, 1, 5)
#             ax_component_1.set_title('Width = {}'.format(widths[0]))
#             ax_component_2.set_title('Width = {}'.format(widths[1]))
#             ax_component_3.set_title('Width = {}'.format(widths[2]))
#             ax_component_4.set_title('Width = {}'.format(widths[3]))
#             # ax_component_5.set_title('Width = {}'.format(width_to_plot[4]))
#             ax_component_1.plot(cwtmatr_1[0, :])
#             ax_component_2.plot(cwtmatr_1[1, :])
#             ax_component_3.plot(cwtmatr_1[2, :])
#             ax_component_4.plot(cwtmatr_1[3, :])
#             # setting
#             plt.subplots_adjust(hspace=0.6)
#
#             fig_cwt_xcor = plt.figure(figsize=(8, 5))
#             fig_cwt_xcor.suptitle('CWT Xcor Result')
#             ax_component_xcor_1 = fig_cwt_xcor.add_subplot(4, 1, 1)
#             ax_component_xcor_2 = fig_cwt_xcor.add_subplot(4, 1, 2)
#             ax_component_xcor_3 = fig_cwt_xcor.add_subplot(4, 1, 3)
#             ax_component_xcor_4 = fig_cwt_xcor.add_subplot(4, 1, 4)
#             # naming
#             ax_component_xcor_1.set_title('Width = {}'.format(widths[0]))
#             ax_component_xcor_2.set_title('Width = {}'.format(widths[1]))
#             ax_component_xcor_3.set_title('Width = {}'.format(widths[2]))
#             ax_component_xcor_4.set_title('Width = {}'.format(widths[3]))
#             # plot
#             ax_component_xcor_1.plot(xcor_map[0, 0, :])
#             ax_component_xcor_2.plot(xcor_map[0, 1, :])
#             ax_component_xcor_3.plot(xcor_map[0, 2, :])
#             ax_component_xcor_4.plot(xcor_map[0, 3, :])
#             # setting
#             plt.subplots_adjust(hspace=0.6)
#
#         fig_xcor = three_dim_visualizer(x_axis=np.arange(1, xcor_map.shape[1] + 1, 1),
#                                         y_axis=widths_2,
#                                         zxx=xcor_map,
#                                         label=['time steps', 'Wavelet Width', 'Correlation Score'],
#                                         output='2d',
#                                         title='CWT Xcor(Normalized+DowSmp) of Sensor[-1m] and Sensor[22m], Source @ {}m'
#                                         .format(p))
#         path = '{}XcorMap_Source @ {}m_DowSmp'.format(savepath, p)
#         fig_xcor.savefig(path)
#         print('Saved !')
#         plt.close()


