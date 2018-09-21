import numpy as np
import pandas as pd
import pywt
import gc
import time
from random import shuffle
import matplotlib.pyplot as plt
from os import listdir
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# self library
from src.utils.helpers import read_all_tdms_from_folder, read_single_tdms, plot_simple_heatmap, \
                              heatmap_visualizer, ProgressBarForLoop, direct_to_dir, shuffle_in_unison
from src.utils.dsp_tools import spectrogram_scipy, butter_bandpass_filtfilt, one_dim_xcor_2d_input, dwt_smoothing, \
    one_dim_xcor_1d_input


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

        # sensor pairing
        # xcor pairing commands - [near] = 0m, 1m,..., 10m
        self.sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
        # xcor pairing commands - [far] = 11m, 12m,..., 20m
        self.sensor_pair_far = [(0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]

    def test_data(self, sensor_dist=None, pressure=None, leak=None):
        '''
        This function returns only a SINGLE tdms file once, for the criteria inputted. This is for testing the algorithm
        without loading tdms in entire folder, which will lead to excessive RAM consumption
        :param sensor_dist: 'near' -> [sensor -3, -2, 2, 4, 6, 8, 10, 12]
                             'far' -> [sensor -3, -2, 10, 14, 16, 18, 20, 22]
                             'plb' -> pencil lead break
        :param pressure: 0 bar, 1 bar or 2 bar
        :param leak: True -> leak, False -> Noleak, 'plb' -> pencil lead break
        :return: raw time series data in2 dimensional array, where axis[0] -> sensors channel
                                                                   axis[1] -> time steps
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

        print('After Swapped Dim: ', n_channel_data.shape)
        return n_channel_data

    def plb(self):
        '''
        This function returns a dataset of xcor map, each belongs to a class of PLB captured by 2 sensors at different
        distance. 51 samples for each class.
        :return:
        Xcor map dataset where axis[0] -> total sample size of all classes
                               axis[1] -> frequency bin
                               axis[2] -> time step (xcor steps)
        '''
        # initialize
        n_channel_data_near = read_all_tdms_from_folder(self.path_plb_2to12)
        n_channel_data_far = read_all_tdms_from_folder(self.path_plb_10to22)
        # take only first 51 samples, so it has same class sizes with data_near
        n_channel_data_far = n_channel_data_far[:51]

        # swap axis, so shape[0] is sensor (for easy using)
        n_channel_data_near = np.swapaxes(n_channel_data_near, 1, 2)  # swap axis[1] and [2]
        n_channel_data_far = np.swapaxes(n_channel_data_far, 1, 2)

        # -----------[BANDPASS + STFT + XCOR]-------------
        # xcor pairing commands - [near] = 0m, 1m,..., 10m
        sensor_pair_near = self.sensor_pair_near
        # invert the sensor pair to generate the opposite lag
        sensor_pair_near_inv = [(pair[1], pair[0]) for pair in sensor_pair_near]

        # xcor pairing commands - [far] = 11m, 12m,..., 20m
        sensor_pair_far = self.sensor_pair_far
        # invert the sensor pair to generate the opposite lag
        sensor_pair_far_inv = [(pair[1], pair[0]) for pair in sensor_pair_far]

        # horizontal segmentation of the xcormap (along xcor step axis)
        xcormap_extent = (250, 550)

        # initialize a dict of lists for every classes
        all_class = {}
        for i in range(-20, 21, 1):
            all_class['class_[{}]'.format(i)] = []

        # class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10 = \
        #     [], [], [], [], [], [], [], [], [], []

        # Sensor [Near] -------------------------------------------------
        # initiate progressbar
        pb = ProgressBarForLoop(title='Bandpass + STFT + XCOR --> [Near]', end=n_channel_data_near.shape[0])
        progress = 0
        # for all plb samples
        for sample in n_channel_data_near:
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

            # xcor for sensor pair (0m) - (10m)
            xcor_map, _ = one_dim_xcor_2d_input(input_mat=all_channel_stft,
                                                pair_list=sensor_pair_near,
                                                verbose=False)

            for i in range(0, 11, 1):
                all_class['class_[{}]'.format(i)].append(xcor_map[i, 10:20, xcormap_extent[0]:xcormap_extent[1]])

            # xcor for sensor pair (0m) - (-10m)
            xcor_map, _ = one_dim_xcor_2d_input(input_mat=all_channel_stft,
                                                pair_list=sensor_pair_near_inv,
                                                verbose=False)
            for i in range(0, 11, 1):
                all_class['class_[{}]'.format(-i)].append(xcor_map[i, 10:20, xcormap_extent[0]:xcormap_extent[1]])

            # update progress bar
            pb.update(now=progress)
            progress += 1
        pb.destroy()

        # shuffle the class_0 and truncate only 51 samples
        shuffle(all_class['class_[0]'])
        all_class['class_[0]'] = all_class['class_[0]'][:51]

        # Sensor [Far] -------------------------------------------------
        # initiate progressbar
        pb = ProgressBarForLoop(title='Bandpass + STFT + XCOR --> [Far]', end=n_channel_data_near.shape[0])
        progress = 0
        for sample in n_channel_data_far:
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

            # horizontal segmentation of the xcormap (along xcor step axis)
            xcormap_extent = (250, 550)

            # xcor for sensor pair (11m) - (20m)
            xcor_map, _ = one_dim_xcor_2d_input(input_mat=all_channel_stft,
                                                pair_list=sensor_pair_far,
                                                verbose=False)

            for i in range(0, 10, 1):
                all_class['class_[{}]'.format(i + 11)].append(xcor_map[i, 10:20, xcormap_extent[0]:xcormap_extent[1]])

            # xcor for sensor pair (11m) - (-20m)
            xcor_map, _ = one_dim_xcor_2d_input(input_mat=all_channel_stft,
                                                pair_list=sensor_pair_far_inv,
                                                verbose=False)

            for i in range(0, 10, 1):
                all_class['class_[{}]'.format(-11-i)].append(xcor_map[i, 10:20, xcormap_extent[0]:xcormap_extent[1]])

            # visualize and saving the training data
            # savepath = 'C:/Users/YH/PycharmProjects/AE-signal-model/result/'
            # visualize = False
            # if visualize:
            #     for i in range(xcor_map.shape[0]):
            #         fig = three_dim_visualizer(x_axis=np.arange(xcormap_extent[0], xcormap_extent[1], 1),
            #                                    y_axis=np.arange(10, 20, 1),
            #                                    zxx=xcor_map[i, 10:20, xcormap_extent[0]:xcormap_extent[1]],
            #                                    output='2d',
            #                                    label=['xcor step', 'freq'],
            #                                    title='({}, {}) = -{}m'.format(sensor_pair_far_inv[i][0],
            #                                                                   sensor_pair_far_inv[i][1],
            #                                                                   i+11))
            #         fig_title = '{}sample{}_xcormap(dist=-{}m)'.format(savepath, progress, i+11)
            #         fig.savefig(fig_title)
            #         plt.close('all')

            # update progress
            pb.update(now=progress)
            progress += 1
        pb.destroy()

        dataset = []
        for i in range(-20, 21, 1):
            temp = np.array(all_class['class_[{}]'.format(i)])
            dataset.append(temp)

        dataset = np.concatenate(dataset, axis=0)
        label = [[i] * 51 for i in np.arange(-20, 21, 1)]
        label = np.array([item for l in label for item in l])

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
                                                 verbose=True)
                all_channel_stft.append(sxx[10:51, :])  # index_10 -> f=20kHz; index_50 -> f=100kHz
            all_channel_stft = np.array(all_channel_stft)

            # xcor for sensor pair
            xcor_map, _ = one_dim_xcor_2d_input(input_mat=all_channel_stft,
                                                pair_list=sensor_pair,
                                                verbose=False)
            # visualize and saving the training data
            savepath = 'C:/Users/YH/PycharmProjects/AE-signal-model/result/'
            visualize = False
            if visualize:
                fig_1 = heatmap_visualizer(x_axis=np.arange(0, xcor_map.shape[2], 1),
                                           y_axis=np.arange(0, xcor_map.shape[1], 1),
                                           zxx=xcor_map[0],
                                           output='2d',
                                           label=['xcor step', 'freq'],
                                           title='(-2, 2)')
                fig_2 = heatmap_visualizer(x_axis=np.arange(0, xcor_map.shape[2], 1),
                                           y_axis=np.arange(0, xcor_map.shape[1], 1),
                                           zxx=xcor_map[1],
                                           output='2d',
                                           label=['xcor step', 'freq'],
                                           title='(-2, 4)')
                fig_3 = heatmap_visualizer(x_axis=np.arange(0, xcor_map.shape[2], 1),
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

    def generate_leak_1bar_in_cwt_xcor_maxpoints_vector(self, saved_filename=None, file_to_process=None, denoise=False):
        '''
        this method read all tdms file from a folder, split each of them into certain parts, perform CWT follow by XCOR
        according to the sensor pair list, then append into a dataset with labels
        :param saved_filename: filename Label for the dataset generated
        :param file_to_process: a list of strings, which is full dir and filename of the tdms to be processed. if none,
                                it is taken as all tdms in the 1bar leak
        :param denoise: True it will denoise the signal bfore CWT and xcor
        :return: dataset where shape[0] -> no of samples of all classes
                               shape[1] -> no of elements in a vector
                 label where shape[0] -> aligned with the shape[0] of dataset
                             shape[1] -> 1
        '''
        # CONFIG -------------------------------------------------------------------------------------------------------
        # DWT
        dwt_wavelet = 'db2'
        dwt_smooth_level = 2

        # CWT
        m_wavelet = 'gaus1'
        scale = np.linspace(2, 10, 100)
        fs = 1e6

        # segmentation per tdms (sample size by each tdms)
        no_of_segment = 2

        # file dir
        if file_to_process is None:
            # list full path of all tdms file in the specified folder
            folder_path = self.path_leak_1bar_2to12
            all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]
        else:
            all_file_path = file_to_process

        # DATA READING -------------------------------------------------------------------------------------------------
        # creating dict to store each class data
        all_class = {}
        for i in range(0, 11, 1):
            all_class['class_[{}]'.format(i)] = []

        # for all tdms file in folder (Warning: It takes 24min for 1 tdms file)
        for tdms_file in all_file_path:

            # read raw from drive
            n_channel_data_near_leak = read_single_tdms(tdms_file)
            n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)

            # split on time axis into no_of_segment
            n_channel_leak = np.split(n_channel_data_near_leak, axis=1, indices_or_sections=no_of_segment)

            if denoise:
                temp = []
                for signal in n_channel_leak:
                    denoised_signal = dwt_smoothing(x=signal, wavelet=dwt_wavelet, level=dwt_smooth_level)
                    temp.append(denoised_signal)
                n_channel_leak = temp

            dist_diff = 0
            # for all sensor combination
            for sensor_pair in self.sensor_pair_near:
                segment_no = 0
                pb = ProgressBarForLoop(title='CWT+Xcor using {}'.format(sensor_pair), end=len(n_channel_leak))
                # for all segmented signals
                for segment in n_channel_leak:
                    pos1_leak_cwt, _ = pywt.cwt(segment[sensor_pair[0]], scales=scale, wavelet=m_wavelet)
                    pos2_leak_cwt, _ = pywt.cwt(segment[sensor_pair[1]], scales=scale, wavelet=m_wavelet)

                    # xcor for every pair of cwt
                    xcor, _ = one_dim_xcor_2d_input(input_mat=np.array([pos1_leak_cwt, pos2_leak_cwt]),
                                                    pair_list=[(0, 1)])
                    xcor = xcor[0]

                    # midpoint in xcor
                    mid = xcor.shape[1] // 2 + 1

                    max_xcor_vector = []
                    # 24000 = fs*24ms(max deviation in ToA)
                    upper_xcor_bound = mid + 24000
                    lower_xcor_bound = mid - 24000
                    # for every row of xcor, find max point index
                    for row in xcor:
                        max_along_x = np.argmax(row[lower_xcor_bound:upper_xcor_bound])
                        max_xcor_vector.append(max_along_x + lower_xcor_bound - mid)

                    # store all feature vector for same class
                    all_class['class_[{}]'.format(dist_diff)].append(max_xcor_vector)

                    # progress
                    pb.update(now=segment_no)
                    segment_no += 1

                    # free up memory for unwanted variable
                    pos1_leak_cwt, pos2_leak_cwt, xcor = None, None, None
                    gc.collect()

                pb.destroy()
                dist_diff += 1

            # just to display the dict full dim
            temp = []
            for _, value in all_class.items():
                temp.append(value[0])
            temp = np.array(temp)
            print('all_class dim: ', temp.shape)

            # free up memory for unwanted variable
            pos1_leak_cwt, pos2_leak_cwt, n_channel_data_near_leak = None, None, None
            gc.collect()

        # transfer all data from dict to array
        dataset = []
        label = []
        # for all class
        for i in range(0, 11, 1):
            # for all samples in a class
            for sample in all_class['class_[{}]'.format(i)]:  # a list of list(max vec)
                dataset.append(sample)
                label.append(i)

        # convert to array
        dataset = np.array(dataset)
        label = np.array(label)
        print('Dataset Dim: ', dataset.shape)
        print('Label Dim: ', label.shape)

        # save to csv
        label = label.reshape((-1, 1))
        all_in_one = np.concatenate([dataset, label], axis=1)
        # column label
        freq = pywt.scale2frequency(wavelet=m_wavelet, scale=scale) * fs
        column_label = ['Scale_{:.4f}_Freq_{:.4f}Hz'.format(i, j) for i, j in zip(scale, freq)] + ['label']
        df = pd.DataFrame(all_in_one, columns=column_label)
        filename = direct_to_dir(where='result') + 'cwt_xcor_maxpoints_vector_dataset_{}.csv'.format(saved_filename)
        df.to_csv(filename)

    def generate_leak_1bar_in_cwt_xcor_maxpoints_vector_2(self, saved_filename=None,
                                                          file_to_process=None, denoise=False):
        '''
        version 2: Instead of cwt for all scale in one shot, we do cwt scale by scale
        this method read all tdms file from a folder, split each of them into certain parts, perform CWT follow by XCOR
        according to the sensor pair list, then append into a dataset with labels
        :param saved_filename: filename Label for the dataset generated
        :param file_to_process: a list of strings, which is full dir and filename of the tdms to be processed. if none,
                                it is taken as all tdms in the 1bar leak
        :param denoise: True it will denoise the signal bfore CWT and xcor
        :return: dataset where shape[0] -> no of samples of all classes
                               shape[1] -> no of elements in a vector
                 label where shape[0] -> aligned with the shape[0] of dataset
                             shape[1] -> 1
        '''
        # CONFIG -------------------------------------------------------------------------------------------------------
        # DWT
        dwt_wavelet = 'db2'
        dwt_smooth_level = 2

        # CWT
        m_wavelet = 'gaus1'
        scale = np.linspace(2, 10, 100)
        fs = 1e6

        # segmentation per tdms (sample size by each tdms)
        no_of_segment = 2

        # file dir
        if file_to_process is None:
            # list full path of all tdms file in the specified folder
            folder_path = self.path_leak_1bar_2to12
            all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]
        else:
            all_file_path = file_to_process

        # DATA READING -------------------------------------------------------------------------------------------------
        # creating dict to store each class data
        all_class = {}
        for i in range(0, 11, 1):
            all_class['class_[{}]'.format(i)] = []

        # for all tdms file in folder (Warning: It takes 24min for 1 tdms file)
        for tdms_file in all_file_path:

            # read raw from drive
            n_channel_data_near_leak = read_single_tdms(tdms_file)
            n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)

            if denoise:
                temp = []
                for signal in n_channel_data_near_leak:
                    denoised_signal = dwt_smoothing(x=signal, wavelet=dwt_wavelet, level=dwt_smooth_level)
                    temp.append(denoised_signal)
                n_channel_data_near_leak = np.array(temp)

            # split on time axis into no_of_segment
            n_channel_leak = np.split(n_channel_data_near_leak, axis=1, indices_or_sections=no_of_segment)

            dist_diff = 0
            # for all sensor combination
            for sensor_pair in self.sensor_pair_near:
                segment_no = 0
                pb = ProgressBarForLoop(title='CWT+Xcor using {}'.format(sensor_pair), end=len(n_channel_leak))
                # for all segmented signals
                for segment in n_channel_leak:

                    max_xcor_vector = []
                    # for all scales
                    for s in scale:
                        pos1_leak_cwt, _ = pywt.cwt(segment[sensor_pair[0]], scales=s, wavelet=m_wavelet)
                        pos2_leak_cwt, _ = pywt.cwt(segment[sensor_pair[1]], scales=s, wavelet=m_wavelet)

                        # xcor for every pair of cwt
                        xcor, _ = one_dim_xcor_1d_input(input_mat=[pos1_leak_cwt, pos2_leak_cwt],
                                                        pair_list=[(0, 1)])
                        xcor = xcor[0]

                        # midpoint in xcor
                        mid = xcor.shape[0] // 2 + 1

                        # 24000 = fs*24ms(max deviation in ToA)
                        upper_xcor_bound = mid + 24000
                        lower_xcor_bound = mid - 24000

                        # for every row of xcor, find max point index
                        max_along_x = np.argmax(xcor[lower_xcor_bound:upper_xcor_bound])
                        max_xcor_vector.append(max_along_x + lower_xcor_bound - mid)

                        # free up memory for unwanted variable
                        pos1_leak_cwt, pos2_leak_cwt, xcor = None, None, None
                        gc.collect()

                    # store all feature vector for same class
                    all_class['class_[{}]'.format(dist_diff)].append(max_xcor_vector)

                    # progress
                    pb.update(now=segment_no)
                    segment_no += 1

                pb.destroy()
                dist_diff += 1

            # just to display the dict full dim
            temp = []
            for _, value in all_class.items():
                temp.append(value[0])
            temp = np.array(temp)
            print('all_class dim: ', temp.shape)

            # free up memory for unwanted variable
            pos1_leak_cwt, pos2_leak_cwt, n_channel_data_near_leak = None, None, None
            gc.collect()

        # transfer all data from dict to array
        dataset = []
        label = []
        # for all class
        for i in range(0, 11, 1):
            # for all samples in a class
            for sample in all_class['class_[{}]'.format(i)]:  # a list of list(max vec)
                dataset.append(sample)
                label.append(i)

        # convert to array
        dataset = np.array(dataset)
        label = np.array(label)
        print('Dataset Dim: ', dataset.shape)
        print('Label Dim: ', label.shape)

        # save to csv
        label = label.reshape((-1, 1))
        all_in_one = np.concatenate([dataset, label], axis=1)
        # column label
        freq = pywt.scale2frequency(wavelet=m_wavelet, scale=scale) * fs
        column_label = ['Scale_{:.4f}_Freq_{:.4f}Hz'.format(i, j) for i, j in zip(scale, freq)] + ['label']
        df = pd.DataFrame(all_in_one, columns=column_label)
        filename = direct_to_dir(where='result') + 'cwt_xcor_maxpoints_vector_dataset_{}.csv'.format(saved_filename)
        df.to_csv(filename)

    def leak_1bar_in_cwt_xcor_maxpoints_vector(self, dataset_name, f_range_to_keep, class_to_keep, shuffle=True):
        '''
        :param dataset_name: name of the csv to be read (it is the name after '...dataset_')
        :param f_range_to_keep: tuple (start, end) where start and end is index along the freq/scale axis
        :param class_to_keep: a list of labels(int), e.g. [0, 2, 3, 5] -> take only those rows with label 0, 2, 3, 5
                              a string 'all' means it will take [0, 1, 2..., 10]
        :param shuffle: this will shuffle so that dataset output is not in order by classes, class0 -> class1 -> class2
        :return: dataset, label
        '''
        # accessing file
        dir = self.path_leak_1bar_2to12 + 'processed/cwt_xcor_maxpoints_vector_dataset_{}.csv'.format(dataset_name)
        data_df = pd.read_csv(dir)
        data_df_col_name = data_df.columns[1:-1]

        # take only certain labels
        if class_to_keep is 'all':
            pass
        elif type(class_to_keep) is list:
            data_df = data_df.loc[data_df['label'].isin(class_to_keep)]

        # convert df values to arrays
        data_mat = data_df.values

        # drop the first column, segment the 2d mat into dataset and label
        dataset = data_mat[:, 1:-1]
        label = data_mat[:, -1]

        # discard the freq range
        if f_range_to_keep is None:
            # take all range
            f_range_to_keep = (0, len(data_df_col_name))
        else:
            dataset = dataset[:, f_range_to_keep[0]:f_range_to_keep[1]]

        # std normalize the data
        dataset_shape = dataset.shape
        scaler = StandardScaler()
        dataset = scaler.fit_transform(dataset.ravel().reshape(-1, 1).astype('float64'))
        dataset = dataset.reshape(dataset_shape)

        # shuffle the data so that output is not class0 -> class1 -> class2
        if shuffle:
            dataset, label = shuffle_in_unison(dataset, label)

        print('Freq Hi: ', data_df_col_name[f_range_to_keep[0]])
        print('Freq Lo: ', data_df_col_name[f_range_to_keep[1]-1])
        print('Dataset Dim: ', dataset.shape)
        print('Label Dim: ', label.shape)

        return dataset, label

    def lcp_recognition_binary_class_dataset(self, train_split, self_shuffle=True):

        # csv file reading
        filename = self.path_leak_1bar_2to12 + 'processed/lcp_recog_1bar_near_segmentation2_dataset.csv'
        # filename = direct_to_dir(where='result') + 'lcp_recog_1bar_near_segmentation2_dataset.csv'
        print('Reading data --> ', filename)
        time_start = time.time()
        data_df = pd.read_csv(filename)
        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Full Dim: ', data_df.values.shape)

        # locate all lcp
        lcp_data = data_df.loc[data_df['label'] == 1].values
        print('LCP Dim: ', lcp_data.shape)

        # normalize the LCP waveform signals (contain +ve and -ve values) into 0-1 only
        lcp_data_only = lcp_data[:, :-1]
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # lcp_data_normalized = scaler.fit_transform(lcp_data_only.ravel().reshape((-1, 1))) \
        #     .reshape((lcp_data_only.shape[0], lcp_data_only.shape[1]))

        # shuffle and split lcp
        lcp_data_train_x, lcp_data_test_x, lcp_data_train_y, lcp_data_test_y = train_test_split(lcp_data_only,
                                                                                                lcp_data[:, -1],
                                                                                                train_size=train_split,
                                                                                                shuffle=self_shuffle)

        non_lcp_data = data_df.loc[data_df['label'] == 0].values
        non_lcp_data_shuffled = non_lcp_data[np.random.permutation(len(non_lcp_data))]
        non_lcp_data_truncated = non_lcp_data_shuffled[:18000]
        print('Non LCP Dim:', non_lcp_data.shape)

        non_lcp_data_train_x, non_lcp_data_test_x, non_lcp_data_train_y, non_lcp_data_test_y = \
            train_test_split(non_lcp_data_truncated[:, :-1],
                             non_lcp_data_truncated[:, -1],
                             train_size=train_split,
                             shuffle=self_shuffle)

        # shuffle non_lcp_data and truncate
        train_x = np.concatenate((non_lcp_data_train_x, lcp_data_train_x))
        test_x = np.concatenate((non_lcp_data_test_x, lcp_data_test_x))
        train_y = np.concatenate((non_lcp_data_train_y, lcp_data_train_y))
        test_y = np.concatenate((non_lcp_data_test_y, lcp_data_test_y))

        print('\n----------TRAIN AND TEST SET---------')
        print('Train_x Dim: ', train_x.shape)
        print('Test_x Dim: ', test_x.shape)
        print('Train_y Dim:', train_y.shape)
        print('Test_y Dim:', test_y.shape)

        return train_x, train_y, test_x, test_y












