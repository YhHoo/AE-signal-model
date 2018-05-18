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

    # 7 position (0-6) | 1-6 are all leak, 0 has leak and no leak
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
        print('Concatenated Data Dim: ', data_0123456.shape)
        print(data_0123456[0, :, 2].shape)

        ft_bank = []
        # for all sensors 0-6
        for i in range(data_0123456.shape[2]):
            for j in range(data_0123456.shape[0]):
                # getting a f-t of dim (5001, 1000)
                ft_mat = spectrogram_scipy(sampled_data=data_0123456[j, :, i],
                                           fs=1e6,
                                           nperseg=10e3,
                                           noverlap=5007,
                                           visualize=False,
                                           verbose=True)
                ft_bank.append(ft_mat)
        ft_bank = np.array(ft_bank)
        print(ft_bank.shape)



    # def noleak_1bar_7pos(self):


data = AccousticEmissionDataSet_16_5_2018()
data.sleak_1bar_7pos()


# print('Reading...', end='')
# tdms_file = TdmsFile('E:/Experiment_16_5_2018/Experiment 1/pos_0_1_2_3/leak/1_bar/set_1.tdms')
# all = tdms_file.as_dataframe()
# print(all.head())
# all_mat = all.values
# print(all_mat)

# if tdms_file.read_completed():
#     print('[Completed]')
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


