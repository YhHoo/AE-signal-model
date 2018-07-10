import numpy as np
import matplotlib.pyplot as plt
# self library
from src.utils.helpers import read_all_tdms_from_folder, three_dim_visualizer, ProgressBarForLoop
from src.utils.dsp_tools import spectrogram_scipy, butter_bandpass_filtfilt, one_dim_xcor_2d_input

class AcousticEmissionDataSet_30_5_2018:
    '''
    The sensor position are (-2, -1, 22, 23)m
    '''
    def __init__(self, drive):
        self.drive = drive + ':/'
        self.path_0m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/0m/PLB/'
        self.path_2m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/2m/PLB/'
        self.path_4m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/4m/PLB/'
        self.path_6m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/6m/PLB/'
        self.path_leak_1bar_10mm = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/leak/10_mm/1_bar/'
        self.path_noleak_1bar = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/no_leak/1_bar/'

    def plb_4_sensor(self, leak_pos=0):
        # ---------------------[Select the file and read]------------------------
        '''
        :param leak_pos: the leak position on the pipe
        :return
        n_channel_data -> 3d matrix where shape[0]-> no of set(sample size),
                                          shape[1]-> no. of AE data points,
                                          shape[2]-> no. of sensors
        phase_map_all -> 4d matrix where shape[0]-> no of set(sample size),
                                         shape[1]-> no. of sensors
                                         shape[2]-> no. of freq band,
                                         shape[3]-> no. of time steps

        '''
        if leak_pos is 0:
            n_channel_data = read_all_tdms_from_folder(self.path_0m_plb)
        elif leak_pos is 2:
            n_channel_data = read_all_tdms_from_folder(self.path_2m_plb)
        elif leak_pos is 4:
            n_channel_data = read_all_tdms_from_folder(self.path_4m_plb)
        elif leak_pos is 6:
            n_channel_data = read_all_tdms_from_folder(self.path_6m_plb)

        # ---------------------[STFT into feature(mag/phase) maps]------------------------
        # for all sets (samples)
        phase_map_all = []
        for set_no in range(n_channel_data.shape[0]):
            phase_map_bank = []
            # for all sensors
            for sensor_no in range(n_channel_data.shape[2]):
                t, f, Sxx, _ = spectrogram_scipy(n_channel_data[set_no, 500000:1500000, sensor_no],
                                                 fs=1e6,
                                                 nperseg=2000,
                                                 noverlap=0,
                                                 mode='magnitude',
                                                 return_plot=False,
                                                 verbose=False,
                                                 vis_max_freq_range=1e6 / 2)
                phase_map_bank.append(Sxx)
            phase_map_bank = np.array(phase_map_bank)
            phase_map_all.append(phase_map_bank)
        # convert to array
        phase_map_all = np.array(phase_map_all)
        print('Phase Map Dim (set_no, sensor_no, freq_band, time steps): ', phase_map_all.shape, '\n')

        return n_channel_data, phase_map_all, f, t

    def leak_noleak_4_sensor(self, leak=True):
        # leak pos at 0m, 1bar, 5mm hole
        # ---------------------[Select the file and read]------------------------
        '''
        :param leak: True is leak, False is no leak
        :return
        n_channel_data -> 3d matrix where shape[0]-> no of set(sample size),
                                          shape[1]-> no. of AE data points,
                                          shape[2]-> no. of sensors
        '''
        if leak is True:
            n_channel_data = read_all_tdms_from_folder(self.path_leak_1bar_10mm)
        else:
            n_channel_data = read_all_tdms_from_folder(self.path_noleak_1bar)

        return n_channel_data

    def test(self):
        n_channel_data = read_all_tdms_from_folder(self.path_leak_1bar_10mm)
        return n_channel_data[0, 0:100000, 0]

    def leak_2class(self):
        '''
        This uses the sensor[-1m] & [23m] as a pair and sensor[-2m] & [22m] as a pair. Both of them has distance of 24m.
        This can act as 2 different leak position, differ by 1m. Since we have only 3 sets of time series, each is 5M
        points, we break them into segments of 100k points (equiv. to 0.1s), so each class will have 150 samples.
        :return:
        class_1 -> 3d array where shape[0]-> no. of samples(150)
                                  shape[1]-> freq axis (20kHz to 100kHz)
                                  shape[2]-> xcor steps ()
        '''
        n_channel_data = read_all_tdms_from_folder(self.path_leak_1bar_10mm)
        # concat all 3 sets into (15M, 4)
        n_channel_data_all_set = np.concatenate((n_channel_data[0], n_channel_data[1], n_channel_data[2]), axis=0)
        # split 15M points into 150 segments
        all_segment_of_4_sensor = np.split(n_channel_data_all_set, indices_or_sections=150)
        all_segment_of_4_sensor = np.array(all_segment_of_4_sensor)

        # STFT for every sample, for every sensors ----------------------------------
        sensor_0_stft, sensor_1_stft, sensor_2_stft, sensor_3_stft = [], [], [], []
        # initiate progressbar
        pb = ProgressBarForLoop(title='Bandpass + STFT', end=all_segment_of_4_sensor.shape[0])
        progress = 0
        # for all samples
        for sample in all_segment_of_4_sensor:
            temp_stft_bank_4_sensor = []
            # for all sensors
            for i in range(4):
                # bandpass from 20kHz t0 100kHz
                filtered_signal = butter_bandpass_filtfilt(sampled_data=sample[:, i],
                                                           fs=1e6,
                                                           f_hicut=1e5,
                                                           f_locut=20e3)
                # STFT
                _, _, sxx, _ = spectrogram_scipy(sampled_data=filtered_signal,
                                                 fs=1e6,
                                                 mode='magnitude',
                                                 nperseg=100,
                                                 nfft=500,
                                                 noverlap=0,
                                                 return_plot=False,
                                                 verbose=False)
                # Take only
                temp_stft_bank_4_sensor.append(sxx[10:51, :])  # index_10 -> f=20kHz; index_50 -> f=100kHz
            sensor_0_stft.append(temp_stft_bank_4_sensor[0])
            sensor_1_stft.append(temp_stft_bank_4_sensor[1])
            sensor_2_stft.append(temp_stft_bank_4_sensor[2])
            sensor_3_stft.append(temp_stft_bank_4_sensor[3])
            # update progressbar
            pb.update(now=progress)
            progress += 1
        pb.destroy()
        sensor_0_stft = np.array(sensor_0_stft)
        sensor_1_stft = np.array(sensor_1_stft)
        sensor_2_stft = np.array(sensor_2_stft)
        sensor_3_stft = np.array(sensor_3_stft)

        # Xcor in freq band
        class_1, class_2 = [], []

        # initiate progressbar
        pb = ProgressBarForLoop(title='Cross Correlation', end=all_segment_of_4_sensor.shape[0])
        progress = 0
        for i in range(150):
            # for class 1, sensor[-2m] & [22m]
            stft_map = np.array([sensor_0_stft[i], sensor_2_stft[i]])
            sensor_pair = [(0, 1)]
            xcor_map = one_dim_xcor_2d_input(input_mat=stft_map,
                                             pair_list=sensor_pair,
                                             verbose=False)
            class_1.append(xcor_map[0, :, 700:1300])

            # for class 1, sensor[-1m] & [23m]
            stft_map = np.array([sensor_1_stft[i], sensor_3_stft[i]])
            sensor_pair = [(0, 1)]
            xcor_map = one_dim_xcor_2d_input(input_mat=stft_map,
                                             pair_list=sensor_pair,
                                             verbose=False)
            class_2.append(xcor_map[0, :, 500:1500])
            pb.update(now=progress)
        pb.destroy()

        # packaging dataset and lable
        class_1 = np.array(class_1)
        class_2 = np.array(class_2)
        dataset = np.concatenate((class_1, class_2), axis=0)
        label = np.array([0] * class_1.shape[0] + [1] * class_2.shape[0])

        print('Dataset Dim: ', dataset.shape)
        print('Label Dim: ', label.shape)

        return dataset, label


# filtered_signal = butter_bandpass_filtfilt(sampled_data=test,
#                                            fs=1e6,
#                                            f_hicut=1e5,
#                                            f_locut=20e3)
#
# t, f, sxx, _ = spectrogram_scipy(sampled_data=test,
#                                  fs=1e6,
#                                  mode='magnitude',
#                                  nperseg=100,
#                                  nfft=500,
#                                  noverlap=0,
#                                  return_plot=False,
#                                  verbose=True)

# fig = three_dim_visualizer(x_axis=np.arange(0, sxx.shape[1], 1),
#                            y_axis=np.arange(0, 51, 1),
#                            zxx=sxx[10:51, :],
#                            output='2d',
#                            label=['timestep', 'freq'])
# plt.show()







