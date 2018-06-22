import numpy as np
from nptdms import TdmsFile
from src.utils.dsp_tools import spectrogram_scipy
from os import listdir
# self defined library
from src.utils.helpers import ProgressBarForLoop, break_into_train_test, read_all_tdms_from_folder


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

    # 7 position (0-6) | 0-6 are 7 leak positions100
    def sleak_1bar_7pos(self, train_split=0.7, f_range=(0, 3000)):
        full_path_0123 = self.drive + self.path_sleak_pos0123_1bar
        full_path_0456 = self.drive + self.path_sleak_pos0456_1bar
        # get all 4 channels sensor data in np matrix of 4 columns
        data_0123 = read_all_tdms_from_folder(full_path_0123)
        data_0456 = read_all_tdms_from_folder(full_path_0456)

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
                                                 return_plot=False,
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
        train_x, test_x, train_y, test_y = break_into_train_test(input=ft_bank,
                                                                 label=ft_bank_label,
                                                                 num_classes=7,
                                                                 train_split=train_split,
                                                                 verbose=True)
        return train_x, test_x, train_y, test_y

    # def noleak_1bar_7pos(self):

#
# data = AccousticEmissionDataSet_16_5_2018()
# data.sleak_1bar_7pos()


