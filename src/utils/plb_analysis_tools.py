import numpy as np
import matplotlib.pyplot as plt
# self lib
from src.utils.dsp_tools import one_dim_xcor_2d_input, spectrogram_scipy
from src.utils.helpers import three_dim_visualizer


def dual_sensor_xcor_with_stft_qiuckview(data_1, data_2, stft_mode, stft_nperseg=500, plot_label=None,
                                         save_selection=[0, 0, 0, 0]):
    '''
    Expecting a time series AE input with 1 PLB peak. Note that the sampling rate is default to 1MHz or 1e6
    :param data_1: 1 dimensional time series sensor data
    :param data_2: 1 dimensional time series sensor data
    :param stft_mode: 'angle', 'phase' or 'magnitude'
    :param stft_nperseg: nperseg actually. noverlap is zero here.
    :param plot_label: list of string, ['leak_position', 'sensor_1_pos', 'sensor_2_pos']
    :param save_selection: a list of 0 n 1 which means dont save n save. The index is equal to the 4 fig in return
    :return:
    fig_time -> time series plot of 2 sensors data
    fig_stft_1 -> spectrogram output of sensor 1 data
    fig_stft_2 -> spectrogram output of sensor 2 data
    fig_xcor -> xcor map color plot where axis[0]-freq band, axis[1]-xcor steps
    '''
    # input data validation
    assert len(data_1) == len(data_2), 'Input data must contains equal no of data points'
    assert len(plot_label) == 3, 'plot_label is incomplete'

    # Time series plot of Data 1 n 2 ---------------------------------------------------------
    fig_time = plt.figure()
    fig_time.subplots_adjust(hspace=0.5)
    # fig 1
    ax1 = fig_time.add_subplot(2, 1, 1)
    ax1.set_title('Time series sensor [{}] @ {}'.format(plot_label[1], plot_label[0]))
    ax2 = fig_time.add_subplot(2, 1, 2)
    ax2.set_title('Time series sensor [{}] @ {}'.format(plot_label[2], plot_label[0]))
    ax1.plot(data_1)
    ax2.plot(data_2)

    # STFT plot of Data 1 n 2 ---------------------------------------------------------------
    _, freq_axis, sxx1, fig_stft_1 = spectrogram_scipy(sampled_data=data_1,
                                                       fs=1e6,
                                                       nperseg=stft_nperseg, nfft=500,
                                                       noverlap=0,
                                                       mode=stft_mode,
                                                       return_plot=True,
                                                       plot_title='Freq-Time rep ({}) of sensor [{}] @ {}'
                                                       .format(stft_mode, plot_label[1], plot_label[0]),
                                                       verbose=False,
                                                       vis_max_freq_range=1e5)  # for now we only visualize up to 100kHz

    _, _, sxx2, fig_stft_2 = spectrogram_scipy(sampled_data=data_2,
                                               fs=1e6,
                                               nperseg=stft_nperseg, nfft=500,
                                               noverlap=0,
                                               mode=stft_mode,
                                               return_plot=True,
                                               plot_title='Freq-Time rep ({}) of sensor [{}] @ {}'
                                               .format(stft_mode, plot_label[2], plot_label[0]),
                                               verbose=False,
                                               vis_max_freq_range=1e5)  # for now we only visualize up to 100kHz

    # Xcor of FT rep of Data 1 n 2 ----------------------------------------------------------
    stft_map = np.array([sxx1, sxx2])
    sensor_pair = [(0, 1)]
    xcor_map = one_dim_xcor_2d_input(input_mat=stft_map,
                                     pair_list=sensor_pair,
                                     verbose=True)
    fig_xcor = three_dim_visualizer(x_axis=np.arange(1, xcor_map.shape[2] + 1, 1),
                                    y_axis=freq_axis,
                                    zxx=xcor_map[0],
                                    label=['Xcor_steps', 'Frequency', 'Correlation Score'],
                                    output='2d', vis_range=[0, 1e5, None, None],
                                    title='PLB {} Map Xcor - Sensor[{}] x Sensor[{}] - @ {}'
                                    .format(stft_mode, plot_label[1], plot_label[2], plot_label[0]))

    # saving 4 fig ---------------------------------------------------------------
    dir = 'C:/Users/YH/PycharmProjects/AE-signal-model/result/'
    if save_selection[0] is 1:
        fig_time.savefig(dir + 'time series @ {}.png'.format(plot_label[0]))
    elif save_selection[1] is 1:
        fig_stft_1.savefig(dir + 'ft map 1 @ {}.png'.format(plot_label[1]))
    elif save_selection[2] is 1:
        fig_stft_2.savefig(dir + 'ft map 2 @ {}.png'.format(plot_label[2]))
    elif save_selection[3] is 1:
        fig_xcor.savefig(dir + 'xcor map @ {}.png'.format(plot_label[0]))

    # plt.close()

    return fig_time, fig_stft_1, fig_stft_2, fig_xcor
