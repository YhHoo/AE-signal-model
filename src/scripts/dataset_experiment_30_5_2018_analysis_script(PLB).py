'''
This script is used to analyze the PLB data sets from experiment 30_5_2018. We aim to finds the recognizable 2d patterns
from the plb time series data, either by STFT in magnitude, phase... and Wavelet.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, ricker
# self lib
from src.experiment_dataset.dataset_experiment_30_5_2018 import AcousticEmissionDataSet_30_5_2018
from src.utils.dsp_tools import one_dim_xcor_freq_band, butter_bandpass_filtfilt
from src.utils.helpers import three_dim_visualizer
from src.utils.plb_analysis_tools import dual_sensor_xcor_with_stft_qiuckview


# -------------------[PLB TEST]-------------------
data = AcousticEmissionDataSet_30_5_2018(drive='F')
set_no = [1, 1, 2, 1]
pos = [0, 2, 4, 6]
# data acquisition for leak pos @ 0m----------------
# n_channel_data, _, _, _ = data.plb_4_sensor(leak_pos=0)
#
# # bandpass from 20kHz to 100kHz
# input_signal_1 = n_channel_data[set_no, 850000:1000000, 1]
# input_signal_2 = n_channel_data[set_no, 850000:1000000, 2]
# input_signal_3 = n_channel_data[set_no, 850000:1000000, 1]
# filtered_signal_1 = butter_bandpass_filtfilt(sampled_data=input_signal_1, fs=1e6, f_hicut=1e5, f_locut=20e3)
# filtered_signal_2 = butter_bandpass_filtfilt(sampled_data=input_signal_2, fs=1e6, f_hicut=1e5, f_locut=20e3)

# -------------------[Xcor testing of spectrogram output]-------------------
stft_analysis = False
if stft_analysis:
    fig1, fig2, fig3, fig4 = dual_sensor_xcor_with_stft_qiuckview(data_1=filtered_signal_1,
                                                                  data_2=filtered_signal_2,
                                                                  stft_mode='magnitude',
                                                                  stft_nperseg=200,
                                                                  plot_label=['0m', '-1m', '22m'],
                                                                  save_selection=[0, 0, 0, 0])


# -------------------[Wavelet Transform]-------------------
widths = np.array([1, 5, 10, 15])
widths_2 = np.arange(1, 20, 0.5)

savepath = 'C:/Users/YH/Desktop/hooyuheng.masterWork/MASTER_PAPERWORK/' \
           'My Practical Work------------/Exp30_5_2018/PLB test/'

for s, p in zip(set_no, pos):
    n_channel_data, _, _, _ = data.plb_4_sensor(leak_pos=p)

    # bandpass from 20kHz to 100kHz
    input_signal_1 = n_channel_data[s, 850000:1000000, 1]
    input_signal_2 = n_channel_data[s, 850000:1000000, 2]
    filtered_signal_1 = butter_bandpass_filtfilt(sampled_data=input_signal_1, fs=1e6, f_hicut=100e3, f_locut=20e3)
    filtered_signal_2 = butter_bandpass_filtfilt(sampled_data=input_signal_2, fs=1e6, f_hicut=100e3, f_locut=20e3)
    cwtmatr_1 = cwt(filtered_signal_1, ricker, widths_2)
    cwtmatr_2 = cwt(filtered_signal_2, ricker, widths_2)
    t = np.arange(850000, 1000000, 1)  # to be defined
    print('CWT output 1 dim: ', cwtmatr_1.shape)
    print('CWT output 2 dim: ', cwtmatr_2.shape)

    fig_cwt_1 = three_dim_visualizer(x_axis=np.arange(1, cwtmatr_1.shape[1] + 1, 1),
                                     y_axis=widths_2,
                                     zxx=cwtmatr_1,
                                     label=['time steps', 'Wavelet Width', 'CWT Coefficient'],
                                     output='2d',
                                     title='CWT Coef of Sensor[-1m], Source @ {}m'.format(p))
    fig_cwt_2 = three_dim_visualizer(x_axis=np.arange(1, cwtmatr_2.shape[1] + 1, 1),
                                     y_axis=widths_2,
                                     zxx=cwtmatr_2,
                                     label=['time steps', 'Wavelet Width', 'CWT Coefficient'],
                                     output='2d',
                                     title='CWT Coef of Sensor[22m], Source @ {}m'.format(p))
    path_s1 = '{}CWT_sensor[{}]_Source@{}m'.format(savepath, '-1m', p)
    path_s2 = '{}CWT_sensor[{}]_Source@{}m'.format(savepath, '22m', p)
    fig_cwt_1.savefig(path_s1)
    fig_cwt_2.savefig(path_s2)
    plt.close()
    print('Saved !')

    cwt_amplitude_plot = False
    if cwt_amplitude_plot:
        cwtmatr_1_max = np.array([i.max() for i in cwtmatr_1])
        cwtmatr_2_max = np.array([i.max() for i in cwtmatr_2])
        fig_cwt_amplitude = plt.figure(figsize=(8, 5))
        ax_cwt_amp_sensor1 = fig_cwt_amplitude.add_subplot(2, 1, 1)
        ax_cwt_amp_sensor2 = fig_cwt_amplitude.add_subplot(2, 1, 2)
        # set title of subplots
        ax_cwt_amp_sensor1.set_title('Maximum Coefficient in CWT of Sensor 1 data against the Widths')
        ax_cwt_amp_sensor2.set_title('Maximum Coefficient in CWT of Sensor 2 data against the Widths')
        # plot
        ax_cwt_amp_sensor1.plot(widths_2, cwtmatr_1_max)
        ax_cwt_amp_sensor2.plot(widths_2, cwtmatr_2_max)

    xcor_in_cwt = True
    if xcor_in_cwt:
        cwt_map = np.array([cwtmatr_1, cwtmatr_2])
        sensor_pair = [(0, 1)]
        xcor_map = one_dim_xcor_freq_band(input_mat=cwt_map,
                                          pair_list=sensor_pair,
                                          verbose=True)
        xcor_map = xcor_map[0, :, 0:xcor_map.shape[2]:10]
        print('DEBUGGING--------->', xcor_map.shape)
        # plotting 4 CWT channels only in a time series plot
        plot_cwt_indi = False
        if plot_cwt_indi:
            # plotting components for CWT xcor result
            fig_cwt_component = plt.figure(figsize=(8, 5))
            fig_cwt_component.suptitle('CWT Components')
            ax_component_1 = fig_cwt_component.add_subplot(4, 1, 1)
            ax_component_2 = fig_cwt_component.add_subplot(4, 1, 2)
            ax_component_3 = fig_cwt_component.add_subplot(4, 1, 3)
            ax_component_4 = fig_cwt_component.add_subplot(4, 1, 4)
            # ax_component_5 = fig_cwt_component.add_subplot(5, 1, 5)
            ax_component_1.set_title('Width = {}'.format(widths[0]))
            ax_component_2.set_title('Width = {}'.format(widths[1]))
            ax_component_3.set_title('Width = {}'.format(widths[2]))
            ax_component_4.set_title('Width = {}'.format(widths[3]))
            # ax_component_5.set_title('Width = {}'.format(width_to_plot[4]))
            ax_component_1.plot(cwtmatr_1[0, :])
            ax_component_2.plot(cwtmatr_1[1, :])
            ax_component_3.plot(cwtmatr_1[2, :])
            ax_component_4.plot(cwtmatr_1[3, :])
            # setting
            plt.subplots_adjust(hspace=0.6)

            fig_cwt_xcor = plt.figure(figsize=(8, 5))
            fig_cwt_xcor.suptitle('CWT Xcor Result')
            ax_component_xcor_1 = fig_cwt_xcor.add_subplot(4, 1, 1)
            ax_component_xcor_2 = fig_cwt_xcor.add_subplot(4, 1, 2)
            ax_component_xcor_3 = fig_cwt_xcor.add_subplot(4, 1, 3)
            ax_component_xcor_4 = fig_cwt_xcor.add_subplot(4, 1, 4)
            # naming
            ax_component_xcor_1.set_title('Width = {}'.format(widths[0]))
            ax_component_xcor_2.set_title('Width = {}'.format(widths[1]))
            ax_component_xcor_3.set_title('Width = {}'.format(widths[2]))
            ax_component_xcor_4.set_title('Width = {}'.format(widths[3]))
            # plot
            ax_component_xcor_1.plot(xcor_map[0, 0, :])
            ax_component_xcor_2.plot(xcor_map[0, 1, :])
            ax_component_xcor_3.plot(xcor_map[0, 2, :])
            ax_component_xcor_4.plot(xcor_map[0, 3, :])
            # setting
            plt.subplots_adjust(hspace=0.6)

        fig_xcor = three_dim_visualizer(x_axis=np.arange(1, xcor_map.shape[1] + 1, 1),
                                        y_axis=widths_2,
                                        zxx=xcor_map,
                                        label=['time steps', 'Wavelet Width', 'Correlation Score'],
                                        output='2d',
                                        title='CWT Xcor(Normalized+DowSmp) of Sensor[-1m] and Sensor[22m], Source @ {}m'
                                        .format(p))
        path = '{}XcorMap_Source @ {}m_DowSmp'.format(savepath, p)
        fig_xcor.savefig(path)
        print('Saved !')
        plt.close()


# ----------------------[Visualize in Time and Saving]----------------------------
time_analysis = False

if time_analysis:
    save_path = 'C:/Users/YH/Desktop/hooyuheng.masterWork/MASTER_PAPERWORK/' \
                'My Practical Work------------/Exp30_5_2018/PLB test/time series/leak @ 6m/'
    # for all sets of the same setup
    for set_no in range(n_channel_data.shape[0]):
        plt.subplot(4, 1, 1)
        plt.plot(n_channel_data[set_no, :, 0])
        plt.title('sensor @ -2m')
        plt.subplot(4, 1, 2)
        plt.plot(n_channel_data[set_no, :, 1])
        plt.title('sensor @ -1m')
        plt.subplot(4, 1, 3)
        plt.plot(n_channel_data[set_no, :, 2])
        plt.title('sensor @ 22m')
        plt.subplot(4, 1, 4)
        plt.plot(n_channel_data[set_no, :, 3])
        plt.title('sensor @ 23m')
        path = save_path + 'set_{}'.format(set_no)
        plt.savefig(path)
        plt.close()
        print('Saved !')

# ----------------------[Xcor for Phase Maps]----------------------------
xcor_analysis = False
save_path = 'C:/Users/YH/Desktop/hooyuheng.masterWork/MASTER_PAPERWORK/' \
            'My Practical Work------------/Exp30_5_2018/PLB test/'

if xcor_analysis:
    # xcor for sensor at -1m and 22m
    sensor_pair = [(0, 1), (1, 2), (0, 3), (2, 3)]
    label = [(-2, -1), (-1, 22), (-2, 23), (22, 23)]

    set_no = 0
    filename = 0
    # for all samples sets
    for set in phase_bank:
        xcor_map = one_dim_xcor_freq_band(input_mat=set,
                                          pair_list=sensor_pair,
                                          verbose=True)
        # max_xscore = []
        # # for all freq bands, take the pos where max xscore happens
        # for row in xcor_map[2]:
        #     max_xscore.append(np.argmax(row))
        # plt.plot(freq_axis, max_xscore, marker='x')
        # plt.show()

        j = 0
        for map in xcor_map:
            fig = three_dim_visualizer(x_axis=np.arange(1, map.shape[1] + 1, 1),
                                       y_axis=freq_axis,
                                       zxx=map,
                                       label=['Xcor_steps', 'Frequency', 'Correlation Score'],
                                       output='2d',
                                       title='PLB Phase Map - Sensor[{}m] x Sensor[{}m] - Set{}'
                                       .format(label[j][0], label[j][1], set_no))

            fig.savefig('C:/Users/YH/Desktop/hooyuheng.masterWork/MASTER_PAPERWORK/'
                        'My Practical Work------------/Exp30_5_2018/PLB test/{}.png'.format(filename))

            print('saved !')
            plt.close()
            filename += 1
            j += 1
        set_no += 1

