import numpy as np
import matplotlib.pyplot as plt
# self library
from src.utils.helpers import read_all_tdms_from_folder, read_single_tdms, multiplot_timeseries, \
                              three_dim_visualizer, ProgressBarForLoop
from src.utils.dsp_tools import spectrogram_scipy, butter_bandpass_filtfilt, one_dim_xcor_2d_input


class AcousticEmissionDataSet_13_7_2018:
    '''
    The sensor position are (-3, -2, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)m
    The leak position is always at 0m, at 10mm diameter
    '''
    def __init__(self, drive):
        self.drive = drive + ':/'
        # Leak (10mm), No Leak n PLB---------------[sensor -3, -2, 2, 4, 6, 8, 10, 12]
        # (0.4 Bar)
        self.path_leak_0bar_2to12 = self.drive \
                                    + 'Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/0.4 bar/Leak/'
        self.path_noleak_0bar_2to12 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/0.4 bar/No_Leak/'
        # (1 Bar)
        self.path_leak_1bar_2to12 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/Leak/'
        self.path_noleak_1bar_2to12 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/No_Leak/'
        # (2 Bar)
        self.path_leak_2bar_2to12 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/2 bar/Leak/'
        self.path_noleak_2bar_2to12 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/2 bar/No_Leak/'
        # PLB (no water flow)
        self.path_plb_2to12 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/PLB/0m/'

        # Leak (10mm), No Leak n PLB---------------[sensor -3, -2, 10, 14, 16, 18, 20, 22]
        # (0.4 Bar)
        self.path_leak_0bar_10to22 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/0.4 bar/Leak/'
        self.path_noleak_0bar_10to22 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/0.4 bar/No_Leak/'
        # (1 Bar)
        self.path_leak_1bar_10to22 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/1 bar/Leak/'
        self.path_noleak_1bar_10to22 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/1 bar/No_Leak/'
        # (2 Bar)
        self.path_leak_2bar_10to22 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/2 bar/Leak/'
        self.path_noleak_2bar_10to22 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/2 bar/No_Leak/'
        # PLB (no water flow)
        self.path_plb_10to22 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/PLB/0m/'

    def test_data(self, sensor_dist=None, pressure=None, leak=None):
        '''
        This function returns only a SINGLE tdms file once, for the criteria inputted. This is for testing the algorithm
        without loading tdms in entire folder, which will lead to excessive RAM consumption
        :param sensor_dist: 'near' -> [sensor -3, -2, 2, 4, 6, 8, 10, 12]
                             'far' -> [sensor -3, -2, 10, 14, 16, 18, 20, 22]
                             'plb' -> pencil lead break
        :param pressure: 0 bar, 1 bar or 2 bar
        :param leak:
        :return:
        '''
        # initialize
        n_channel_data = None
        # sensor [2-12] or [10-22]
        if sensor_dist is 'near':
            if pressure is 0 and leak is True:
                n_channel_data = read_single_tdms(self.path_leak_0bar_2to12 + 'test data/test.tdms')
            elif pressure is 0 and leak is False:
                n_channel_data = read_single_tdms(self.path_noleak_0bar_2to12 + 'test data/test.tdms')
            elif pressure is 1 and leak is True:
                n_channel_data = read_single_tdms(self.path_leak_1bar_2to12 + 'test data/test.tdms')
            elif pressure is 1 and leak is False:
                n_channel_data = read_single_tdms(self.path_noleak_1bar_2to12 + 'test data/test.tdms')
            elif pressure is 2 and leak is True:
                n_channel_data = read_single_tdms(self.path_leak_2bar_2to12 + 'test data/test.tdms')
            elif pressure is 2 and leak is False:
                n_channel_data = read_single_tdms(self.path_noleak_2bar_2to12 + 'test data/test.tdms')
            elif leak is 'plb':
                n_channel_data = read_single_tdms(self.path_plb_2to12 + 'test_001.tdms')
        elif sensor_dist is 'far':
            if pressure is 0 and leak is True:
                n_channel_data = read_single_tdms(self.path_leak_0bar_10to22 + 'test data/test.tdms')
            elif pressure is 0 and leak is False:
                n_channel_data = read_single_tdms(self.path_noleak_0bar_10to22 + 'test data/test.tdms')
            elif pressure is 1 and leak is True:
                n_channel_data = read_single_tdms(self.path_leak_1bar_10to22 + 'test data/test.tdms')
            elif pressure is 1 and leak is False:
                n_channel_data = read_single_tdms(self.path_noleak_1bar_10to22 + 'test data/test.tdms')
            elif pressure is 2 and leak is True:
                n_channel_data = read_single_tdms(self.path_leak_2bar_10to22 + 'test data/test2.tdms')
            elif pressure is 2 and leak is False:
                n_channel_data = read_single_tdms(self.path_noleak_2bar_10to22 + 'test data/test.tdms')
            elif leak is 'plb':
                n_channel_data = read_single_tdms(self.path_plb_10to22 + 'test_001.tdms')

        # swap axis, so shape[0] is sensor (for easy using)
        n_channel_data = np.swapaxes(n_channel_data, 0, 1)  # swap axis[0] and [1]
        return n_channel_data

    def plb(self, sensor_dist=None):
        '''
        This function returns a dataset of xcor map, each belongs to a class of PLB captured by 2 sensors at different
        distance.
        :return:
        Xcor map
        '''
        # initialize
        n_channel_data = None
        if sensor_dist is 'near':
            n_channel_data = read_all_tdms_from_folder(self.path_plb_2to12)
        elif sensor_dist is 'far':
            n_channel_data = read_all_tdms_from_folder(self.path_plb_10to22)
        elif sensor_dist is 'all':
            data_near = read_all_tdms_from_folder(self.path_plb_2to12)
            data_far = read_all_tdms_from_folder(self.path_plb_10to22)
            n_channel_data = np.concatenate((data_near, data_far))

        # swap axis, so shape[0] is sensor (for easy using)
        n_channel_data = np.swapaxes(n_channel_data, 1, 2)  # swap axis[1] and [2]

        # -----------[BANDPASS + STFT + XCOR]-------------
        sensor_pair = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
        class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10, class_11 = \
            [], [], [], [], [], [], [], [], [], [], []

        # initiate progressbar
        pb = ProgressBarForLoop(title='Bandpass + STFT + XCOR', end=n_channel_data.shape[0])
        progress = 0
        # for all plb samples
        for sample in n_channel_data:
            all_channel_stft = []
            # for all sensors
            for each_sensor_data in sample:
                # band pass filter
                filtered_signal = butter_bandpass_filtfilt(sampled_data=each_sensor_data[90000:130000],
                                                           fs=1e6,
                                                           f_hicut=1e5,
                                                           f_locut=20e3)
                # stft
                _, _, sxx, _ = spectrogram_scipy(sampled_data=filtered_signal,
                                                 fs=1e6,
                                                 mode='magnitude',
                                                 nperseg=100,
                                                 nfft=500,
                                                 noverlap=0,
                                                 return_plot=False,
                                                 verbose=False)
                all_channel_stft.append(sxx[10:51, :])  # index_10 -> f=20kHz; index_50 -> f=100kHz
            all_channel_stft = np.array(all_channel_stft)

            # xcor for sensor pair
            xcor_map = one_dim_xcor_2d_input(input_mat=all_channel_stft,
                                             pair_list=sensor_pair,
                                             verbose=False)
            # visualize and saving the training data
            savepath = 'C:/Users/YH/PycharmProjects/AE-signal-model/result/'
            visualize = False
            if visualize:
                for i in range(xcor_map.shape[0]):
                    fig = three_dim_visualizer(x_axis=np.arange(300, 500, 1),
                                               y_axis=np.arange(10, 20, 1),
                                               zxx=xcor_map[i, 10:20, 300:500],
                                               output='2d',
                                               label=['xcor step', 'freq'],
                                               title='({}, {}) = {}m'.format(sensor_pair[i][0], sensor_pair[i][1], i))
                    fig_title = '{}sample{}_xcormap(dist={}m)'.format(savepath, progress, i)
                    fig.savefig(fig_title)
                    plt.close('all')

            class_1.append(xcor_map[0, 10:20, 300:500])
            class_2.append(xcor_map[1, 10:20, 300:500])
            class_3.append(xcor_map[2, 10:20, 300:500])
            class_4.append(xcor_map[3, 10:20, 300:500])
            class_5.append(xcor_map[4, 10:20, 300:500])
            class_6.append(xcor_map[5, 10:20, 300:500])
            class_7.append(xcor_map[6, 10:20, 300:500])
            class_8.append(xcor_map[7, 10:20, 300:500])
            class_9.append(xcor_map[8, 10:20, 300:500])
            class_10.append(xcor_map[9, 10:20, 300:500])
            class_11.append(xcor_map[10, 10:20, 300:500])
            # update progress
            pb.update(now=progress)
            progress += 1
        pb.destroy()

        # packaging and labeling dataset
        class_1 = np.array(class_1)
        class_2 = np.array(class_2)
        class_3 = np.array(class_3)
        class_4 = np.array(class_4)
        class_5 = np.array(class_5)
        class_6 = np.array(class_6)
        class_7 = np.array(class_7)
        class_8 = np.array(class_8)
        class_9 = np.array(class_9)
        class_10 = np.array(class_10)
        class_11 = np.array(class_11)
        dataset = np.concatenate((class_1, class_2, class_3, class_4, class_5,
                                  class_6, class_7, class_8, class_9, class_10, class_11), axis=0)
        label = np.array([0] * class_1.shape[0] + [1] * class_2.shape[0] +
                         [2] * class_3.shape[0] + [3] * class_4.shape[0] +
                         [4] * class_5.shape[0] + [5] * class_6.shape[0] +
                         [6] * class_7.shape[0] + [7] * class_8.shape[0] +
                         [8] * class_9.shape[0] + [9] * class_10.shape[0] +
                         [10] * class_11.shape[0])

        print('Dataset Dim: ', dataset.shape)
        print('Label Dim: ', label.shape)

        return dataset, label

    def plb_unseen(self):
        # use near only
        n_channel_data = read_all_tdms_from_folder(self.path_plb_2to12)

        # swap axis, so shape[0] is sensor (for easy using)
        n_channel_data = np.swapaxes(n_channel_data, 1, 2)  # swap axis[0] and [1]

        # pairing order for xcor
        sensor_pair = [(0, 5)]
        class_1 = []
        # initiate progressbar
        pb = ProgressBarForLoop(title='Bandpass + STFT + XCOR', end=n_channel_data.shape[0])
        progress = 0
        # for all plb samples
        for sample in n_channel_data:
            all_channel_stft = []
            # for all sensors
            for each_sensor_data in sample:
                # band pass filter
                filtered_signal = butter_bandpass_filtfilt(sampled_data=each_sensor_data[90000:130000],
                                                           fs=1e6,
                                                           f_hicut=1e5,
                                                           f_locut=20e3)
                # stft
                _, _, sxx, _ = spectrogram_scipy(sampled_data=filtered_signal,
                                                 fs=1e6,
                                                 mode='magnitude',
                                                 nperseg=100,
                                                 nfft=500,
                                                 noverlap=0,
                                                 return_plot=False,
                                                 verbose=False)
                all_channel_stft.append(sxx[10:51, :])  # index_10 -> f=20kHz; index_50 -> f=100kHz
            all_channel_stft = np.array(all_channel_stft)

            # xcor for sensor pair
            xcor_map = one_dim_xcor_2d_input(input_mat=all_channel_stft,
                                             pair_list=sensor_pair,
                                             verbose=False)
            # visualize and saving the training data
            savepath = 'C:/Users/YH/PycharmProjects/AE-signal-model/result/'
            visualize = False
            if visualize:
                fig_1 = three_dim_visualizer(x_axis=np.arange(0, xcor_map.shape[2], 1),
                                             y_axis=np.arange(0, xcor_map.shape[1], 1),
                                             zxx=xcor_map[0],
                                             output='2d',
                                             label=['xcor step', 'freq'],
                                             title='(-2, 2)')
                fig_2 = three_dim_visualizer(x_axis=np.arange(0, xcor_map.shape[2], 1),
                                             y_axis=np.arange(0, xcor_map.shape[1], 1),
                                             zxx=xcor_map[1],
                                             output='2d',
                                             label=['xcor step', 'freq'],
                                             title='(-2, 4)')
                fig_3 = three_dim_visualizer(x_axis=np.arange(0, xcor_map.shape[2], 1),
                                             y_axis=np.arange(0, xcor_map.shape[1], 1),
                                             zxx=xcor_map[2],
                                             output='2d',
                                             label=['xcor step', 'freq'],
                                             title='(-2, 6)')

                fig_1_title = '{}sample{}_xcormap(-2, 2)'.format(savepath, progress)
                fig_2_title = '{}sample{}_xcormap(-2, 4)'.format(savepath, progress)
                fig_3_title = '{}sample{}_xcormap(-2, 6)'.format(savepath, progress)

                fig_1.savefig(fig_1_title)
                fig_2.savefig(fig_2_title)
                fig_3.savefig(fig_3_title)

                plt.close('all')

            class_1.append(xcor_map[0, 10:20, 300:500])

            # update progress
            pb.update(now=progress)
            progress += 1
        pb.destroy()

        class_1 = np.array(class_1)
        print('Dataset Dim: ', class_1.shape)

        return class_1

    def plb_raw(self, sensor_dist=None):
        # initialize
        n_channel_data = None
        if sensor_dist is 'near':
            n_channel_data = read_all_tdms_from_folder(self.path_plb_2to12)
        elif sensor_dist is 'far':
            n_channel_data = read_all_tdms_from_folder(self.path_plb_10to22)

        # swap axis, so shape[0] is sensor (for easy using)
        n_channel_data = np.swapaxes(n_channel_data, 1, 2)  # swap axis[0] and [1]

        return n_channel_data


# data = AcousticEmissionDataSet_13_7_2018(drive='F')
# data_raw = data.plb(sensor_dist='near')


# -------------[VISUALIZE ALL IN TIME]---------------
# subplot_titles = ['sensor[{}m]'.format(d) for d in [-3, -2, 10, 14, 16, 18, 20, 22]]
# sample_no = 0
# savepath = 'C:/Users/YH/PycharmProjects/AE-signal-model/result/'
# for sample in data_raw:
#     fig = multiplot_timeseries(sample[:, 90000:130000],
#                                subplot_titles=subplot_titles,
#                                main_title='PLB time series (sample_{})'.format(sample_no))
#     filename = '{}PLB_time_ser_sample{}'.format(savepath, sample_no)
#     fig.savefig(filename)
#     plt.close('all')
#     print('saved')
#     sample_no += 1








