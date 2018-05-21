import numpy as np
from nptdms import TdmsFile
from dsp_tools import spectrogram_scipy, fft_scipy
from os import listdir
# self defined library
from utils import ProgressBarForLoop


class AccousticEmissionDataSet_16_5_2018:
    def __init__(self):
        print('----------AE DATA SET RAW [16-5-2018]---------')
        self.drive = 'E:/'
        # directory
        # Pressure = 1 bar | sleak = 0.5cm Hole | distance = 0-6m
        self.path_sleak_pos0123_1bar = 'Experiment_16_5_2018/Experiment 1/pos_0_1_2_3/leak/1_bar/'
        self.path_sleak_pos0456_1bar = 'Experiment_16_5_2018/Experiment 1/pos_0_4_5_6/leak/1_bar/'
        self.path_noleak_pos0123_1bar = 'Experiment_16_5_2018/Experiment 1/pos_0_1_2_3/no_leak/1_bar/'
        self.path_noleak_pos0456_1bar = 'Experiment_16_5_2018/Experiment 1/pos_0_4_5_6/no_leak/1_bar/'
        self.path_sleak_pos0123_2bar = 'Experiment_16_5_2018/Experiment 1/pos_0_1_2_3/leak/2_bar/'
        self.path_sleak_pos0456_2bar = 'Experiment_16_5_2018/Experiment 1/pos_0_4_5_6/leak/2_bar/'
        self.path_noleak_pos0123_2bar = 'Experiment_16_5_2018/Experiment 1/pos_0_1_2_3/no_leak/2_bar/'
        self.path_noleak_pos0456_2bar = 'Experiment_16_5_2018/Experiment 1/pos_0_4_5_6/no_leak/2_bar/'

    # private non-callable function, only used in this class to read all file of same kind
    @staticmethod
    def _read_tdms_from_folder(folder_path=None):
        '''
        :param folder_path: The folder which contains several sets data of same setup (Test rig)
        :return: 3d matrix where shape[0]=no. of sets | shape[1]=no. of AE Signal points | shape[2]=no. of sensors
        Aim: To combine all sets of data for same experiment setup into one 3d array.
        WARNING: All sets of input data must contains same number of points e.g. 5 seconds/5M points for all sets
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
            # store the df values to list
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
    def _break_into_train_test(input, label, num_classes, train_split=0.7, verbose=False):
        '''
        :param input: expect a 3d np array where 1st index is total sample size
        :param label: expect a 1d np array of same size as input.shape[0]
        :param num_classes: total classes to break into
        :param verbose: print the summary of train test size
        :return: a train and test set
        AIM----------------------------------
        This is when we receive a list of N classes samples all concatenate together sequentially
        e.g [0,..,0,1,..1,2,..,2...N-1..N-1] and we want to split them into train and test.
        WARNING------------------------------
        Every class size have to be EQUAL !
        '''
        # ensure both input and label sample size are equal
        assert input.shape[0] == label.shape[0], 'Sample size of Input and Label must be equal !'
        print('\n----------TRAIN AND TEST SET---------')
        sample_size = input.shape[0]
        # create an index where the
        class_break_index = np.linspace(0, sample_size, num_classes + 1)
        # convert from float to int
        class_break_index = [int(i) for i in class_break_index]
        # determine split index from first 2 items of class_break_index list
        split_index_from_start = int(train_split * (class_break_index[1] - class_break_index[0]))

        # training set
        train_x, test_x, train_y, test_y = [], [], [], []
        # slicing in btw every intervals for classes
        for i in range(len(class_break_index) - 1):
            train_x.append(input[class_break_index[i]: (class_break_index[i] + split_index_from_start)])
            test_x.append(input[(class_break_index[i] + split_index_from_start): class_break_index[i+1]])
            train_y.append(label[class_break_index[i]: (class_break_index[i] + split_index_from_start)])
            test_y.append(label[(class_break_index[i] + split_index_from_start): class_break_index[i+1]])

        # convert list of list into just a list
        train_x = [data for classes in train_x for data in classes]
        test_x = [data for classes in test_x for data in classes]
        train_y = [data for classes in train_y for data in classes]
        test_y = [data for classes in test_y for data in classes]

        # convert list to np array
        train_x = np.array(train_x)
        test_x = np.array(test_x)
        train_y = np.array(train_y)
        test_y = np.array(test_y)

        if verbose:
            print('Split Index from start: ', split_index_from_start)
            print('Train_x Dim: ', train_x.shape)
            print('Test_x Dim: ', test_x.shape)
            print('Train_y Dim:', train_y.shape)
            print('Test_y Dim:', test_y.shape)

        # return
        return train_x, train_y, test_x, test_y

    # 7 position (0-6) | 0-6 are 7 leak positions100
    def sleak_1bar_7pos(self, train_split=0.7, f_range=(0, 3000)):
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

        # FREQ-TIME TRANSFORMATION
        ft_bank = []
        ft_bank_label = []
        label = 0
        # for all 7 sensors (0-6)
        pb = ProgressBarForLoop('Spectrogram_scipy Transformation', end=data_0123456.shape[2])
        for i in range(data_0123456.shape[2]):
            pb.update(now=i)
            # for all 3 sets
            for j in range(data_0123456.shape[0]):
                # getting a f-t of dim (5001, 1000) for a 5 seconds signal
                _, _, ft_mat = spectrogram_scipy(sampled_data=data_0123456[j, :, i],
                                                 fs=1e6,
                                                 nperseg=10000,
                                                 noverlap=5007,
                                                 visualize=False,
                                                 verbose=False,
                                                 save=False,
                                                 save_title='Class[{}] Set[{}]'.format(i, j))
                # split 5 sec into 25x0.2sec for bigger sample size
                # take only 0 - 300kHz (3000 points) out of 0 - 500kHz (5001 points)
                index_start = 0
                index_end = ft_mat.shape[1]  # time step
                interval = 25  # how many parts u wan to split into
                # plus 1 is necessary because N points only has N-1 intervals
                segmented_index = np.linspace(index_start, index_end, interval + 1)
                # convert all in segmented_index to int
                segmented_index = [int(i) for i in segmented_index]
                # for all interval in segmented_index
                for k in range(len(segmented_index)-1):
                    ft_bank.append(ft_mat[f_range[0]:f_range[1], segmented_index[k]:segmented_index[k+1]])
                    ft_bank_label.append(label)
            # label for nex position
            label += 1
        # kill progress bar
        pb.destroy()
        ft_bank = np.array(ft_bank)
        ft_bank_label = np.array(ft_bank_label)
        print('f-t Data Dim: ', ft_bank.shape)
        print('Label Dim:', ft_bank_label.shape)

        # slicing them into train and test set
        train_x, test_x, train_y, test_y = self._break_into_train_test(input=ft_bank,
                                                                       label=ft_bank_label,
                                                                       num_classes=7,
                                                                       train_split=train_split,
                                                                       verbose=True)
        return train_x, test_x, train_y, test_y

    # def noleak_1bar_7pos(self):

#
# data = AccousticEmissionDataSet_16_5_2018()
# data.sleak_1bar_7pos()


