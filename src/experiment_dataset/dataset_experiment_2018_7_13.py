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
        This function returns only a single tdms file once, for the criteria inputted. This is for testing the algorithm
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

        # swap axis, so shape[0] is sensor (for easy using)
        n_channel_data = np.swapaxes(n_channel_data, 1, 2)  # swap axis[0] and [1]

        # initiate progressbar
        pb = ProgressBarForLoop(title='Bandpass + STFT', end=n_channel_data.shape[0])
        progress = 0

        # BANDPASS + STFT
        stft_bank = []
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
            stft_bank.append(all_channel_stft)
            # update progress
            pb.update(now=progress)
            progress += 1
        pb.destroy()
        stft_bank = np.array(stft_bank)
        print(stft_bank.shape)


data = AcousticEmissionDataSet_13_7_2018(drive='F')
data.plb(sensor_dist='near')

# plb_8_channel = data.test_data(sensor_dist='near', leak='plb')
# # so that axis[0] equal to sensors
# plb_8_channel = np.swapaxes(plb_8_channel, 0, 1)
# # titles setting
# # subplot_title = ['sensor {}m'.format(s_no) for s_no in [-3, -2, 2, 4, 6, 8, 10, 12]]
#
# stft_map_8_sensors = []
# # for all sensor data
# for raw_data in plb_8_channel:
#     # band pass filter
#     filtered_signal = butter_bandpass_filtfilt(sampled_data=raw_data, fs=1e6, f_hicut=1e5, f_locut=20e3)
#
#     # stft
#     _, _, sxx, _ = spectrogram_scipy(sampled_data=filtered_signal[90000:130000],
#                                      fs=1e6,
#                                      mode='magnitude',
#                                      nperseg=100,
#                                      nfft=500,
#                                      noverlap=0,
#                                      return_plot=False,
#                                      verbose=False)
#
#     stft_map_8_sensors.append(sxx[10:51, :])  # index_10 -> f=20kHz; index_50 -> f=100kHz
#
# stft_map_8_sensors = np.array(stft_map_8_sensors)

# sensor_pair = [(1, 2), (1, 3), (1, 7)]
# xcor_map = one_dim_xcor_2d_input(input_mat=stft_map_8_sensors,
#                                  pair_list=sensor_pair,
#                                  verbose=True)
#
# class_1_xcor_map = xcor_map[0]
# class_2_xcor_map = xcor_map[1]
# class_3_xcor_map = xcor_map[2]
#
# fig1 = three_dim_visualizer(x_axis=np.arange(0, class_1_xcor_map.shape[1], 1),
#                             y_axis=np.arange(0, 41, 1),
#                             zxx=class_1_xcor_map,
#                             output='2d',
#                             label=['timestep', 'freq'],
#                             title='(-2, 2)')
# fig2 = three_dim_visualizer(x_axis=np.arange(0, class_2_xcor_map.shape[1], 1),
#                             y_axis=np.arange(0, 41, 1),
#                             zxx=class_2_xcor_map,
#                             output='2d',
#                             label=['timestep', 'freq'],
#                             title='(-2, 10)')
# fig3 = three_dim_visualizer(x_axis=np.arange(0, class_3_xcor_map.shape[1], 1),
#                             y_axis=np.arange(0, 41, 1),
#                             zxx=class_3_xcor_map,
#                             output='2d',
#                             label=['timestep', 'freq'],
#                             title='(-2, 12)')

# plt.show()

