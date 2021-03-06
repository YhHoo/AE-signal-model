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
        self.random_leak_dataset_filename = self.drive + 'Experiment_3_10_2018/LCP x NonLCP DATASET/' \
                                                         'dataset_leak_random_1bar_3_norm.csv'
        self.random_noleak_dataset_filename = self.drive + 'Experiment_3_10_2018/LCP x NonLCP DATASET/' \
                                                           'dataset_noleak_random_2bar_norm.csv'

    def lcp_dataset_binary_class(self, shuffle_b4_split=True, train_split=0.7):
        '''
        Dataset of peaks, where 1 refers to peak during leakage, 0 refers to peak during no leak
        :param shuffle_b4_split:
        :param train_split:
        :return:
        '''
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

    def lcp_by_distance_dataset_multi_class(self, shuffle_b4_split=True, train_split=0.7):
        '''
        This is aim to train a model that feed on the data from sensor, and it is able to tel from how far the LCP comes
        from.
        the dataset is normalized LCP, labeled according to distance travelled away from source.
        So, class -2m and 2m will be classied as same, because both LCP travels same dist to reach the sensor
        dist = 2, 4.5, 5, 8, 10m, labeled as 0, 1, 2, 3, 4
        '''

        # reading lcp data fr csv
        time_start = time.time()
        print('Reading --> ', self.lcp_dataset_filename)
        df_lcp = pd.read_csv(self.lcp_dataset_filename)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('LCP Dataset Dim: ', df_lcp.values.shape)

        print('Reading --> ', self.non_lcp_dataset_filename)
        df_non_lcp = pd.read_csv(self.non_lcp_dataset_filename)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('non LCP Dataset Dim: ', df_non_lcp.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for non LCP -------------------------------------------------------------------------------------------------
        non_lcp_data = df_non_lcp.values[:, :-1]
        if shuffle_b4_split:
            non_lcp_data = non_lcp_data[np.random.permutation(len(non_lcp_data))]

        label = [0] * len(non_lcp_data)

        tr_x, te_x, tr_y, te_y = train_test_split(non_lcp_data,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for -2 & 2m --------------------------------------------------------------------------------------------------
        mat_of_selected_ch = df_lcp.loc[df_lcp['channel'].isin([1, 2])].values[:, :-1]
        if shuffle_b4_split:
            mat_of_selected_ch = mat_of_selected_ch[np.random.permutation(len(mat_of_selected_ch))]

        label = [1] * len(mat_of_selected_ch)

        tr_x, te_x, tr_y, te_y = train_test_split(mat_of_selected_ch,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for -4.5m ----------------------------------------------------------------------------------------------------
        mat_of_selected_ch = df_lcp[df_lcp['channel'] == 0].values[:, :-1]
        if shuffle_b4_split:
            mat_of_selected_ch = mat_of_selected_ch[np.random.permutation(len(mat_of_selected_ch))]

        label = [2] * len(mat_of_selected_ch)

        tr_x, te_x, tr_y, te_y = train_test_split(mat_of_selected_ch,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for 5, 8, 10m ------------------------------------------------------------------------------------------------
        for ch_no in [3, 4, 5]:
            mat_of_selected_ch = df_lcp[df_lcp['channel'] == ch_no].values[:, :-1]

            if shuffle_b4_split:
                mat_of_selected_ch = mat_of_selected_ch[np.random.permutation(len(mat_of_selected_ch))]

            label = [ch_no] * len(mat_of_selected_ch)

            tr_x, te_x, tr_y, te_y = train_test_split(mat_of_selected_ch,
                                                      label,
                                                      train_size=train_split,
                                                      shuffle=True)

            train_x.append(tr_x)
            test_x.append(te_x)
            train_y.append(tr_y)
            test_y.append(te_y)

        train_x = np.concatenate(train_x, axis=0)
        test_x = np.concatenate(test_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        test_y = np.concatenate(test_y, axis=0)

        print('\n----------TRAIN AND TEST SET---------')
        print('Train_x Dim: ', train_x.shape)
        print('Test_x Dim: ', test_x.shape)
        print('Train_y Dim:', train_y.shape)
        print('Test_y Dim:', test_y.shape)

        return train_x, train_y, test_x, test_y

    def random_leak_noleak_by_dist_dataset_multiclass(self, shuffle_b4_split=True, train_split=0.7):
        # reading lcp data fr csv
        time_start = time.time()
        print('Reading --> ', self.random_leak_dataset_filename)
        df_leak_rand = pd.read_csv(self.random_leak_dataset_filename)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak Dataset Dim: ', df_leak_rand.values.shape)

        print('Reading --> ', self.random_noleak_dataset_filename)
        df_noleak_rand = pd.read_csv(self.random_noleak_dataset_filename)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random no leak Dataset Dim: ', df_noleak_rand.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for no leak random -------------------------------------------------------------------------------------------
        noleak_data = df_noleak_rand.values[:, :-1]
        if shuffle_b4_split:
            noleak_data = noleak_data[np.random.permutation(len(noleak_data))]

        label = [0] * len(noleak_data)

        tr_x, te_x, tr_y, te_y = train_test_split(noleak_data,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for -2 & 2m --------------------------------------------------------------------------------------------------
        mat_of_selected_ch = df_leak_rand.loc[df_leak_rand['channel'].isin([1, 2])].values[:, :-1]
        if shuffle_b4_split:
            mat_of_selected_ch = mat_of_selected_ch[np.random.permutation(len(mat_of_selected_ch))]

        label = [1] * len(mat_of_selected_ch)

        tr_x, te_x, tr_y, te_y = train_test_split(mat_of_selected_ch,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for -4.5m ----------------------------------------------------------------------------------------------------
        mat_of_selected_ch = df_leak_rand.loc[df_leak_rand['channel'] == 0].values[:, :-1]
        if shuffle_b4_split:
            mat_of_selected_ch = mat_of_selected_ch[np.random.permutation(len(mat_of_selected_ch))]

        label = [2] * len(mat_of_selected_ch)

        tr_x, te_x, tr_y, te_y = train_test_split(mat_of_selected_ch,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for 5, 8, 10m ------------------------------------------------------------------------------------------------
        for ch_no in [3, 4, 5]:
            mat_of_selected_ch = df_leak_rand.loc[df_leak_rand['channel'] == ch_no].values[:, :-1]

            if shuffle_b4_split:
                mat_of_selected_ch = mat_of_selected_ch[np.random.permutation(len(mat_of_selected_ch))]

            label = [ch_no] * len(mat_of_selected_ch)

            tr_x, te_x, tr_y, te_y = train_test_split(mat_of_selected_ch,
                                                      label,
                                                      train_size=train_split,
                                                      shuffle=True)

            train_x.append(tr_x)
            test_x.append(te_x)
            train_y.append(tr_y)
            test_y.append(te_y)

        train_x = np.concatenate(train_x, axis=0)
        test_x = np.concatenate(test_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        test_y = np.concatenate(test_y, axis=0)

        print('\n----------TRAIN AND TEST SET---------')
        print('Train_x Dim: ', train_x.shape)
        print('Test_x Dim: ', test_x.shape)
        print('Train_y Dim:', train_y.shape)
        print('Test_y Dim:', test_y.shape)

        return train_x, train_y, test_x, test_y


# data = AcousticEmissionDataSet_3_10_2018(drive='F')
# data.random_leak_noleak_by_dist_dataset_multiclass()
# train_x, train_y, test_x, test_y = data.lcp_by_distance_dataset_multi_class()
#
#
# print(test_y[:100])
# print(test_y[-1000:])
