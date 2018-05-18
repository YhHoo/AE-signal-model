import numpy as np
from nptdms import TdmsFile
from dsp_tools import spectrogram_scipy, fft_scipy
from os import listdir
# self defined library
from utils import ProgressBarForLoop


class AccousticEmissionDataSet_16_5_2018:
    def __init__(self):
        print('----------AE DATA SET RAW [16-5-2018]---------')
        self.drive = 'F:/'
        # directory
        # Pressure = 1 bar | sleak = 0.5cm Hole | distance = 0-6m
        self.path_sleak_pos0123_1bar = 'Experiment_16_5_2018/Experiment 1/pos_0_1_2_3/leak/1_bar/'
        self.path_sleak_pos0456_1bar = 'Experiment_16_5_2018/Experiment 1/pos_0_4_5_6/leak/1_bar/'
        self.path_noleak_pos0123_1bar = 'Experiment_16_5_2018/Experiment 1/pos_0_1_2_3/no_leak/1_bar/'
        self.path_noleak_pos0456_1bar = 'Experiment_16_5_2018/Experiment 1/pos_0_4_5_6/no_leak/1_bar/'

    # private non-callable function, only used in this class to read all file of same kind
    @staticmethod
    def _read_tdms_from_folder(folder_path=None):
        '''
        :param folder_path: The folder which contains several sets data of same setup (Test rig)
        :return: 3d matrix where shape[0]=no. of sets | shape[1]=no. of AE Signal points | shape[2]=no. of sensors
        Aim: To combine all sets of data for same experiment setup into one 3d array.
        '''
        # ensure path exist
        assert folder_path is not None, 'No Folder is selected'

        # list full path of all tdms file in the specified folder
        all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]
        n_channel_matrix = []
        # do for all 3 sets of tdms file
        # read tdms and save as 4 channel np array
        pb = ProgressBarForLoop('Reading <-- ' + folder_path, end=len(all_file_path))
        for f in all_file_path:
            tdms_file = TdmsFile(f)
            tdms_df = tdms_file.as_dataframe()
            n_channel_matrix.append(tdms_df.values)
            # update progress
            pb.update(now=all_file_path.index(f))
        # kill progress bar
        pb.destroy()
        # convert the list matrix
        n_channel_matrix = np.array(n_channel_matrix)
        print('Data Dim: ', n_channel_matrix.shape, '\n')

        return n_channel_matrix

    @staticmethod
    def _break_into_train_test(input, num_classes, train_split=0.7, verbose=False):
        '''
        :param input: expect a 3d np array where 1st index is total sample size
        :param num_classes: total classes to break into
        :param verbose: print the summary of train test size
        :return: a train and test set
        aim: This is when we receive a list of N classes samples all concatenate together sequentially and
        we want to split them into train and test.
        '''
        print('\n----------TRAIN AND TEST SET---------')
        sample_size = input.shape[0]
        # create an index where the
        class_break_index = np.linspace(0, sample_size, num_classes + 1)
        # convert from float to int
        class_break_index = [int(i) for i in class_break_index]
        # determine split index from first 2 items of class_break_index list
        split_index_from_start = int(train_split * (class_break_index[1] - class_break_index[0]))

        # training set
        train_x, test_x = [], []
        # slicing in btw every intervals for classes
        for i in range(len(class_break_index) - 1):
            train_x.append(input[class_break_index[i]: (class_break_index[i] + split_index_from_start)])
            test_x.append(input[(class_break_index[i] + split_index_from_start): class_break_index[i+1]])
        # define train and test exact sizes
        train_size, test_size = len(train_x[0]), len(test_x[0])
        # convert list of list into just a list
        train_x = [data for classes in train_x for data in classes]
        test_x = [data for classes in test_x for data in classes]
        # convert list to np array
        train_x = np.array(train_x)
        test_x = np.array(test_x)

        # labels
        train_y, test_y = [], []
        for i in np.arange(0, 7, 1):
            train_y.append([i] * train_size)
            test_y.append([i] * test_size)
        train_y = [i for sublist in train_y for i in sublist]
        test_y = [i for sublist in test_y for i in sublist]

        if verbose:
            print('Split Index from start: ', split_index_from_start)
            print('Train_x Dim: ', train_x.shape)
            print('Test_x Dim: ', test_x.shape)
            print('Train_y Dim:', len(train_y))
            print('Test_y Dim:', len(test_y))

        # return
        return train_x, train_y, test_x, test_y

    # 7 position (0-6) | 0-6 are 7 leak positions100
    def sleak_1bar_7pos(self):
        full_path_0123 = self.drive + self.path_sleak_pos0123_1bar
        full_path_0456 = self.drive + self.path_sleak_pos0456_1bar
        # get all 4 channels sensor data in np matrix of 4 columns
        data_0123 = self._read_tdms_from_folder(full_path_0123)
        data_0456 = self._read_tdms_from_folder(full_path_0456)
        # ensure both has same number of sets before concatenate
        assert data_0123.shape[0] == data_0456.shape[0], 'Different no of sets detected'
        data_0123456 = []
        for i in range(data_0123.shape[0]):
            # take only column 1-3 for data_0456
            data_0123456.append(np.concatenate((data_0123[i], data_0456[i, :, 1:]), axis=1))
        data_0123456 = np.array(data_0123456)
        print('Concatenated Data Dim: ', data_0123456.shape, '\n')

        ft_bank = []
        # for all sensors 0-6
        pb = ProgressBarForLoop('Spectrogram_scipy Transformation', end=data_0123456.shape[2])
        for i in range(data_0123456.shape[2]):
            pb.update(now=i)
            for j in range(data_0123456.shape[0]):
                # getting a f-t of dim (5001, 1000)
                _, _, ft_mat = spectrogram_scipy(sampled_data=data_0123456[j, :, i],
                                                 fs=1e6,
                                                 nperseg=10000,
                                                 noverlap=5007,
                                                 visualize=False,
                                                 verbose=False)
                # split 5 sec into 25x0.2sec for bigger sample size
                # take only 0 - 300kHz out of 0 - 500kHz
                index_start = 0
                index_end = 1000
                interval = 25
                # plus 1 is necessary because N points only has N-1 intervals
                segmented_index = np.linspace(index_start, index_end, interval + 1)
                # convert all in segmented_index to int
                segmented_index = [int(i) for i in segmented_index]
                for k in range(len(segmented_index)-1):
                    ft_bank.append(ft_mat[:3000, segmented_index[k]:segmented_index[k+1]])
        # kill progress bar
        pb.destroy()
        ft_bank = np.array(ft_bank)
        print('f-t Data Dim: ', ft_bank.shape)

        # slicing them into train and test set
        train_x, test_x, train_y, test_y = self._break_into_train_test(ft_bank,
                                                                       num_classes=7,
                                                                       train_split=0.7,
                                                                       verbose=True)
        return train_x, test_x, train_y, test_y

    # def noleak_1bar_7pos(self):

data = AccousticEmissionDataSet_16_5_2018()
data.sleak_1bar_7pos()


# print('Reading...', end='')
# tdms_file = TdmsFile('F:/Experiment_16_5_2018/Experiment 1/pos_0_1_2_3/leak/1_bar/set_1.tdms')
# all = tdms_file.as_dataframe()
# print(all.head())
# all_mat = all.values
# print(all_mat)

# voltage_0 = tdms_file.object('Untitled', 'Voltage_0')
# voltage_0_data = voltage_0.data
# print(voltage_0_data.shape)
#
# spectrogram_scipy(sampled_data=voltage_0_data,
#                   fs=1e6,
#                   nperseg=10000,
#                   noverlap=5007,
#                   visualize=False,
#                   verbose=True)


