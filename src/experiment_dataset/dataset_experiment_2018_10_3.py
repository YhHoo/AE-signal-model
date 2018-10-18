import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class AcousticEmissionDataSet_3_10_2018:
    '''
        The sensor position are (-4.5, -2, 2, 5, 8, 10)m
        The leak position is always at 0m, at 10mm diameter. The LCP and nonLCP is collected at 1bar.
        raw data -> find_peak_for_all_script() -> data_preparation_script() -> .csv
    '''
    def __init__(self, drive):
        self.drive = drive + ':/'
        self.lcp_dataset_filename = self.drive + 'Experiment_3_10_2018/LCP x NonLCP DATASET/' \
                                                 'dataset_lcp_1bar_seg4_norm.csv'
        self.non_lcp_dataset_filename = self.drive + 'Experiment_3_10_2018/LCP x NonLCP DATASET/' \
                                                     'dataset_non_lcp_1bar_seg1_norm.csv'

    def lcp_dataset_binary_class(self, shuffle_b4_split=True, train_split=0.7):

        # reading lcp data fr csv
        time_start = time.time()
        print('Reading --> ', self.lcp_dataset_filename)
        df_lcp = pd.read_csv(self.lcp_dataset_filename)

        print('Reading --> ', self.non_lcp_dataset_filename)
        df_non_lcp = pd.read_csv(self.non_lcp_dataset_filename)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('LCP Dataset Dim: ', df_lcp.values.shape)
        print('non LCP Dataset Dim: ', df_non_lcp.values.shape)

        # data slicing
        lcp_data = df_lcp.values[:, :-1]
        lcp_data_with_label = np.concatenate((lcp_data, np.ones((len(lcp_data), 1))), axis=1)

        non_lcp_data = df_non_lcp.values[:, :-1]
        non_lcp_data_with_label = np.concatenate((non_lcp_data, np.zeros((len(non_lcp_data), 1))), axis=1)

        # shuffle the data before test train splitting
        if shuffle_b4_split:
            non_lcp_data_with_label = non_lcp_data_with_label[np.random.permutation(len(non_lcp_data_with_label))]
            lcp_data_with_label = lcp_data_with_label[np.random.permutation(len(lcp_data_with_label))]

        # train test split
        lcp_train_x, lcp_test_x, lcp_train_y, lcp_test_y = \
            train_test_split(lcp_data_with_label[:, :-1],
                             lcp_data_with_label[:, -1],
                             train_size=train_split,
                             shuffle=True)

        non_lcp_train_x, non_lcp_test_x, non_lcp_train_y, non_lcp_test_y = \
            train_test_split(non_lcp_data_with_label[:, :-1],
                             non_lcp_data_with_label[:, -1],
                             train_size=train_split,
                             shuffle=True)

        # concate 2 lcp and non-lcp classes
        train_x = np.concatenate((non_lcp_train_x, lcp_train_x))
        test_x = np.concatenate((non_lcp_test_x, lcp_test_x))
        train_y = np.concatenate((non_lcp_train_y, lcp_train_y))
        test_y = np.concatenate((non_lcp_test_y, lcp_test_y))

        print('\n----------TRAIN AND TEST SET---------')
        print('Train_x Dim: ', train_x.shape)
        print('Test_x Dim: ', test_x.shape)
        print('Train_y Dim:', train_y.shape)
        print('Test_y Dim:', test_y.shape)

        return train_x, train_y, test_x, test_y


# data = AcousticEmissionDataSet_3_10_2018(drive='F')
# _, _, _, _ = data.lcp_dataset_binary_class(train_split=0.7)
