import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class AcousticEmissionDataSet:
    '''
        The sensor position are (-4.5, -2, 2, 5, 8, 10)m
        The leak position is always at 0m, at 10mm diameter.
    '''
    def __init__(self, drive):
        self.drive = drive + ':/'

        # LCP and NON LCP
        self.lcp_dataset_filename = self.drive + 'Experiment_3_10_2018/LCP x NonLCP DATASET/' \
                                                 'dataset_lcp_1bar_seg4_norm.csv'
        self.non_lcp_dataset_filename = self.drive + 'Experiment_3_10_2018/LCP x NonLCP DATASET/' \
                                                     'dataset_non_lcp_1bar_seg1_norm.csv'

        # NORMALIZED ---------------------------------------------------------------------------------------------------
        # RANDOM LEAK NO LEAK (21 December data)
        self.random_leak_2bar_dec_p1 = self.drive + 'Experiment_21_12_2018/leak_noleak_preprocessed_dataset/' \
                                                    'dataset_leak_random_2bar_[-4,-2,2,4,6,8,10]_norm.csv'
        self.random_noleak_2bar_dec_p1 = self.drive + 'Experiment_21_12_2018/leak_noleak_preprocessed_dataset/' \
                                                      'dataset_noleak_random_2bar_[-4,-2,2,6,8,10]_norm.csv'
        # sensor @ 0m
        self.random_leak_2bar_dec_p2 = self.drive + 'Experiment_21_12_2018/leak_noleak_preprocessed_dataset/' \
                                                    'dataset_leak_random_2bar_[0]_norm.csv'
        self.random_noleak_2bar_dec_p2 = self.drive + 'Experiment_21_12_2018/leak_noleak_preprocessed_dataset/' \
                                                      'dataset_noleak_random_2bar_[0]_norm.csv'

        # RANDOM NO LEAK (13 July data)
        # sensor @ 4m
        self.random_noleak_2bar_july = self.drive + 'Experiment_13_7_2018/Experiment 1/' \
                                                    'leak_noleak_preprocessed_dataset/' \
                                                    'dataset_noleak_random_2bar_[4]_norm.csv'

        # No NORMALIZED ------------------------------------------------------------------------------------------------
        # RANDOM LEAK NO LEAK (21 December data)
        self.random_leak_2bar_dec_p1_xn = self.drive + 'Experiment_21_12_2018/leak_noleak_preprocessed_dataset/' \
                                                    'dataset_leak_random_2bar_[-4,-2,2,4,6,8,10].csv'
        self.random_noleak_2bar_dec_p1_xn = self.drive + 'Experiment_21_12_2018/leak_noleak_preprocessed_dataset/' \
                                                      'dataset_noleak_random_2bar_[-4,-2,2,6,8,10].csv'
        # sensor @ 0m
        self.random_leak_2bar_dec_p2_xn = self.drive + 'Experiment_21_12_2018/leak_noleak_preprocessed_dataset/' \
                                                    'dataset_leak_random_2bar_[0].csv'
        self.random_noleak_2bar_dec_p2_xn = self.drive + 'Experiment_21_12_2018/leak_noleak_preprocessed_dataset/' \
                                                      'dataset_noleak_random_2bar_[0].csv'

        # RANDOM NO LEAK (13 July data)
        # sensor @ 4m
        self.random_noleak_2bar_july_xn = self.drive + 'Experiment_13_7_2018/Experiment 1/' \
                                                    'leak_noleak_preprocessed_dataset/' \
                                                    'dataset_noleak_random_2bar_[4].csv'

    def random_leak_noleak_dec_july(self, shuffle_b4_split=True, train_split=0.7):
        # reading lcp data fr csv

        # sensor @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.random_leak_2bar_dec_p1)
        df_leak_rand = pd.read_csv(self.random_leak_2bar_dec_p1)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak Dataset Dim: ', df_leak_rand.values.shape)

        # sensor @ -4, -2, 2, 6, 8, 10m
        print('Reading --> ', self.random_noleak_2bar_dec_p1)
        df_noleak_rand = pd.read_csv(self.random_noleak_2bar_dec_p1)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random no leak Dataset Dim: ', df_noleak_rand.values.shape)

        # sensor @ 0m
        time_start = time.time()
        print('Reading --> ', self.random_leak_2bar_dec_p2)
        df_leak_rand_2 = pd.read_csv(self.random_leak_2bar_dec_p2)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak Dataset Dim: ', df_leak_rand_2.values.shape)

        print('Reading --> ', self.random_noleak_2bar_dec_p2)
        df_noleak_rand_2 = pd.read_csv(self.random_noleak_2bar_dec_p2)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random no leak Dataset Dim: ', df_noleak_rand_2.values.shape)

        # sensor @ 4m
        print('Reading --> ', self.random_noleak_2bar_july)
        df_noleak_rand_3 = pd.read_csv(self.random_noleak_2bar_july)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random no leak Dataset Dim: ', df_noleak_rand_3.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for no leak random -------------------------------------------------------------------------------------------
        # sensor @ -4, -2, 2, 6, 8, 10m ---
        noleak_data = df_noleak_rand.values[:, :-1]
        if shuffle_b4_split:
            noleak_data = noleak_data[np.random.permutation(len(noleak_data))]

        # noleak_data = noleak_data[:7000, :]

        # sensor @ 0m ---
        noleak_data_2 = df_noleak_rand_2.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_2 = noleak_data_2[np.random.permutation(len(noleak_data_2))]

        # noleak_data_2 = noleak_data_2[:2000, :]

        # sensor @ 4m ---
        noleak_data_3 = df_noleak_rand_3.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_3 = noleak_data_3[np.random.permutation(len(noleak_data_3))]

        # noleak_data_3 = noleak_data_3[:2000, :]

        # merging all noleak scenario
        label = [0] * (len(noleak_data) + len(noleak_data_2) + len(noleak_data_3))
        noleak_all = np.concatenate((noleak_data, noleak_data_2, noleak_data_3), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(noleak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for leak random -------------------------------------------------------------------------------------------
        # sensor @ -4, -2, 2, 4, 6, 8, 10m ---
        leak_data = df_leak_rand.values[:, :-1]
        if shuffle_b4_split:
            leak_data = leak_data[np.random.permutation(len(leak_data))]

        # leak_data = leak_data[:7000, :]

        # sensor @ 0m ---
        leak_data_2 = df_leak_rand_2.values[:, :-1]
        if shuffle_b4_split:
            leak_data_2 = leak_data_2[np.random.permutation(len(leak_data_2))]

        # leak_data_2 = leak_data_2[:2000, :]

        label = [1] * (len(leak_data) + len(leak_data_2))

        # merging all noleak scenario
        leak_all = np.concatenate((leak_data, leak_data_2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
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

    def random_leak_noleak_dec_july_no_norm(self, shuffle_b4_split=True, train_split=0.7):
        # reading lcp data fr csv

        # sensor @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.random_leak_2bar_dec_p1_xn)
        df_leak_rand = pd.read_csv(self.random_leak_2bar_dec_p1_xn)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak Dataset Dim: ', df_leak_rand.values.shape)

        # sensor @ -4, -2, 2, 6, 8, 10m
        print('Reading --> ', self.random_noleak_2bar_dec_p1_xn)
        df_noleak_rand = pd.read_csv(self.random_noleak_2bar_dec_p1_xn)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random no leak Dataset Dim: ', df_noleak_rand.values.shape)

        # sensor @ 0m
        time_start = time.time()
        print('Reading --> ', self.random_leak_2bar_dec_p2_xn)
        df_leak_rand_2 = pd.read_csv(self.random_leak_2bar_dec_p2_xn)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak Dataset Dim: ', df_leak_rand_2.values.shape)

        print('Reading --> ', self.random_noleak_2bar_dec_p2_xn)
        df_noleak_rand_2 = pd.read_csv(self.random_noleak_2bar_dec_p2_xn)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random no leak Dataset Dim: ', df_noleak_rand_2.values.shape)

        # sensor @ 4m
        print('Reading --> ', self.random_noleak_2bar_july_xn)
        df_noleak_rand_3 = pd.read_csv(self.random_noleak_2bar_july_xn)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random no leak Dataset Dim: ', df_noleak_rand_3.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for no leak random -------------------------------------------------------------------------------------------
        # sensor @ -4, -2, 2, 6, 8, 10m ---
        noleak_data = df_noleak_rand.values[:, :-1]
        if shuffle_b4_split:
            noleak_data = noleak_data[np.random.permutation(len(noleak_data))]

        # noleak_data = noleak_data[:7000, :]

        # sensor @ 0m ---
        noleak_data_2 = df_noleak_rand_2.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_2 = noleak_data_2[np.random.permutation(len(noleak_data_2))]

        # noleak_data_2 = noleak_data_2[:2000, :]

        # sensor @ 4m ---
        noleak_data_3 = df_noleak_rand_3.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_3 = noleak_data_3[np.random.permutation(len(noleak_data_3))]

        # noleak_data_3 = noleak_data_3[:2000, :]

        # merging all noleak scenario
        label = [0] * (len(noleak_data) + len(noleak_data_2) + len(noleak_data_3))
        noleak_all = np.concatenate((noleak_data, noleak_data_2, noleak_data_3), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(noleak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for leak random -------------------------------------------------------------------------------------------
        # sensor @ -4, -2, 2, 4, 6, 8, 10m ---
        leak_data = df_leak_rand.values[:, :-1]
        if shuffle_b4_split:
            leak_data = leak_data[np.random.permutation(len(leak_data))]

        # leak_data = leak_data[:7000, :]

        # sensor @ 0m ---
        leak_data_2 = df_leak_rand_2.values[:, :-1]
        if shuffle_b4_split:
            leak_data_2 = leak_data_2[np.random.permutation(len(leak_data_2))]

        # leak_data_2 = leak_data_2[:2000, :]

        label = [1] * (len(leak_data) + len(leak_data_2))

        # merging all noleak scenario
        leak_all = np.concatenate((leak_data, leak_data_2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
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

    def random_leak_noleak_by_dist_dec_july(self, shuffle_b4_split=True, train_split=0.7):
        # reading lcp data fr csv

        # sensor @ -4, -2, 2, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.random_leak_2bar_dec_p1)
        df_leak_rand = pd.read_csv(self.random_leak_2bar_dec_p1)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak Dataset Dim: ', df_leak_rand.values.shape)

        print('Reading --> ', self.random_noleak_2bar_dec_p1)
        df_noleak_rand = pd.read_csv(self.random_noleak_2bar_dec_p1)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random no leak Dataset Dim: ', df_noleak_rand.values.shape)

        # sensor @ 0m
        time_start = time.time()
        print('Reading --> ', self.random_leak_2bar_dec_p2)
        df_leak_rand_2 = pd.read_csv(self.random_leak_2bar_dec_p2)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak Dataset Dim: ', df_leak_rand_2.values.shape)

        print('Reading --> ', self.random_noleak_2bar_dec_p2)
        df_noleak_rand_2 = pd.read_csv(self.random_noleak_2bar_dec_p2)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random no leak Dataset Dim: ', df_noleak_rand_2.values.shape)

        # sensor @ 4m
        print('Reading --> ', self.random_noleak_2bar_july)
        df_noleak_rand_3 = pd.read_csv(self.random_noleak_2bar_july)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random no leak Dataset Dim: ', df_noleak_rand_3.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for no leak random -------------------------------------------------------------------------------------------
        # sensor @ -4, -2, 2, 6, 8, 10m ---
        noleak_data = df_noleak_rand.values[:, :-1]
        if shuffle_b4_split:
            noleak_data = noleak_data[np.random.permutation(len(noleak_data))]

        noleak_data = noleak_data[:7000, :]

        # sensor @ 0m ---
        noleak_data_2 = df_noleak_rand_2.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_2 = noleak_data_2[np.random.permutation(len(noleak_data_2))]

        noleak_data_2 = noleak_data_2[:2000, :]

        # sensor @ 4m ---
        noleak_data_3 = df_noleak_rand_3.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_3 = noleak_data_3[np.random.permutation(len(noleak_data_3))]

        noleak_data_3 = noleak_data_3[:2000, :]

        # merging all noleak scenario
        label = [0] * (len(noleak_data) + len(noleak_data_2) + len(noleak_data_3))
        noleak_all = np.concatenate((noleak_data, noleak_data_2, noleak_data_3), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(noleak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for 0m ----------------------------------------------------------------------------------------------------
        mat_of_selected_ch = df_leak_rand_2.loc[df_leak_rand_2['channel'] == 0].values[:, :-1]
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

        # for -2 & 2m --------------------------------------------------------------------------------------------------
        mat_of_selected_ch = df_leak_rand.loc[df_leak_rand['channel'].isin([1, 2])].values[:, :-1]
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

        # for -4 & 4m --------------------------------------------------------------------------------------------------
        mat_of_selected_ch = df_leak_rand.loc[df_leak_rand['channel'].isin([0, 3])].values[:, :-1]
        if shuffle_b4_split:
            mat_of_selected_ch = mat_of_selected_ch[np.random.permutation(len(mat_of_selected_ch))]

        label = [3] * len(mat_of_selected_ch)

        tr_x, te_x, tr_y, te_y = train_test_split(mat_of_selected_ch,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for 6, 8, 10m ------------------------------------------------------------------------------------------------
        for ch_no in [4, 5, 6]:
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


# data = AcousticEmissionDataSet(drive='F')
# data_random = data.random_leak_noleak_dec_july()


