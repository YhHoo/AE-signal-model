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

        # No Normalized (6000 points per sample) -----------------------------------------------------------------------
        # leak
        self.leak_dataset_filename_p1 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                     'dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10].csv'
        self.leak_dataset_filename_p2 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                     'dataset_leak_random_1.5bar_[0].csv'

        # no leak
        self.noleak_dataset_filename_p1 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                       'dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10].csv'
        self.noleak_dataset_filename_p2 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                       'dataset_noleak_random_1.5bar_[0].csv'

        # No Normalized & Downsampled to fs=200kHz (2000 points per sample) --------------------------------------------
        # seen leak
        self.leak_dataset_filename_p1_ds = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                     'dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds.csv'
        self.leak_dataset_filename_p2_ds = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                     'dataset_leak_random_1.5bar_[0]_ds.csv'

        # seen no leak
        self.noleak_dataset_filename_p1_ds = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                       'dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds.csv'
        self.noleak_dataset_filename_p2_ds = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                       'dataset_noleak_random_1.5bar_[0]_ds.csv'

        # unseen leak
        self.leak_dataset_filename_p3_ds = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                        'dataset_leak_random_1.5bar_[-3,5,7,16,17]_ds.csv'
        # unseen no leak
        self.noleak_dataset_filename_p3_ds = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_noleak_random_1.5bar_[-3,5,7,16,17]_ds.csv'

        # No Normalized & Downsampled to fs=100kHz (2000 points per sample) --------------------------------------------
        # Seen leak
        self.leak_dataset_filename_p1_ds2 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                        'dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds2.csv'
        self.leak_dataset_filename_p2_ds2 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                        'dataset_leak_random_1.5bar_[0]_ds2.csv'

        # Seen no leak
        self.noleak_dataset_filename_p1_ds2 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                          'dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds2.csv'
        self.noleak_dataset_filename_p2_ds2 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                          'dataset_noleak_random_1.5bar_[0]_ds2.csv'

        # unseen leak
        self.leak_dataset_filename_p3_ds2 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                        'dataset_leak_random_1.5bar_[-3,5,7,16,17]_ds2.csv'
        # unseen no leak
        self.noleak_dataset_filename_p3_ds2 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                          'dataset_noleak_random_1.5bar_[-3,5,7,16,17]_ds2.csv'

        # No Normalized & Downsampled to fs=50kHz (2000 points per sample) --------------------------------------------
        # Seen leak
        self.leak_dataset_filename_p1_ds3 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds3.csv'
        self.leak_dataset_filename_p2_ds3 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_leak_random_1.5bar_[0]_ds3.csv'

        # Seen no leak
        self.noleak_dataset_filename_p1_ds3 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                           'dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds3.csv'
        self.noleak_dataset_filename_p2_ds3 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                           'dataset_noleak_random_1.5bar_[0]_ds3.csv'

        # unseen leak
        self.leak_dataset_filename_p3_ds3 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_leak_random_1.5bar_[-3,5,7,16,17]_ds3.csv'
        # unseen no leak
        self.noleak_dataset_filename_p3_ds3 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                           'dataset_noleak_random_1.5bar_[-3,5,7,16,17]_ds3.csv'

        # No Normalized & Downsampled to fs=25kHz (2000 points per sample) --------------------------------------------
        # Seen leak
        self.leak_dataset_filename_p1_ds4 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds4.csv'
        self.leak_dataset_filename_p2_ds4 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_leak_random_1.5bar_[0]_ds4.csv'

        # Seen no leak
        self.noleak_dataset_filename_p1_ds4 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                           'dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds4.csv'
        self.noleak_dataset_filename_p2_ds4 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                           'dataset_noleak_random_1.5bar_[0]_ds4.csv'

        # unseen leak
        self.leak_dataset_filename_p3_ds4 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_leak_random_1.5bar_[-3,5,7,16,17]_ds4.csv'
        # unseen no leak
        self.noleak_dataset_filename_p3_ds4 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                           'dataset_noleak_random_1.5bar_[-3,5,7,16,17]_ds4.csv'

        # No Normalized & Downsampled to fs=2kHz (2000 points per sample) --------------------------------------------
        # Seen leak
        self.leak_dataset_filename_p1_ds5 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds5.csv'
        self.leak_dataset_filename_p2_ds5 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_leak_random_1.5bar_[0]_ds5.csv'

        # Seen no leak
        self.noleak_dataset_filename_p1_ds5 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                           'dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds5.csv'
        self.noleak_dataset_filename_p2_ds5 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                           'dataset_noleak_random_1.5bar_[0]_ds5.csv'

        # unseen leak
        self.leak_dataset_filename_p3_ds5 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_leak_random_1.5bar_[-3,5,7,16,17]_ds5.csv'
        # unseen no leak
        self.noleak_dataset_filename_p3_ds5 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                           'dataset_noleak_random_1.5bar_[-3,5,7,16,17]_ds5.csv'

        # No Normalized & Downsampled to fs=1kHz (2000 points per sample) --------------------------------------------
        # Seen leak
        self.leak_dataset_filename_p1_ds6 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds6.csv'
        self.leak_dataset_filename_p2_ds6 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_leak_random_1.5bar_[0]_ds6.csv'

        # Seen no leak
        self.noleak_dataset_filename_p1_ds6 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                           'dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds6.csv'
        self.noleak_dataset_filename_p2_ds6 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                           'dataset_noleak_random_1.5bar_[0]_ds6.csv'

        # unseen leak
        self.leak_dataset_filename_p3_ds6 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                         'dataset_leak_random_1.5bar_[-3,5,7,16,17]_ds6.csv'
        # unseen no leak
        self.noleak_dataset_filename_p3_ds6 = self.drive + 'Experiment_3_1_2019/leak_noleak_preprocessed_dataset/' \
                                                           'dataset_noleak_random_1.5bar_[-3,5,7,16,17]_ds6.csv'


    def random_leak_noleak(self, shuffle_b4_split=True, train_split=0.7):
        '''
        The leak of p1 contains 109 tdms files, and noleak of p1 only contains 49 tdms files. So we wil half the samples
        size from leak p1.
        The leak of p2 also contains 117 tdms files and noleak of p2 only containes 57 tdms files. to ensure balance
        class in training, half the sample size of leak p2.

        Also, remove the signals from channel -4m and -2m. we only want to train the model on recognizing leak no leak
        from 0m up to 10m, assuming the leak is always a one side of the sensor.
        '''
        # read leak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p1)
        df_leak_rand_p1 = pd.read_csv(self.leak_dataset_filename_p1)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p1 Dataset Dim: ', df_leak_rand_p1.values.shape)

        # read leak @ 0m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p2)
        df_leak_rand_p2 = pd.read_csv(self.leak_dataset_filename_p2)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p2 Dataset Dim: ', df_leak_rand_p2.values.shape)

        # read noleak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p1)
        df_noleak_rand_p1 = pd.read_csv(self.noleak_dataset_filename_p1)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p1 Dataset Dim: ', df_noleak_rand_p1.values.shape)

        # read noleak @ 0m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p2)
        df_noleak_rand_p2 = pd.read_csv(self.noleak_dataset_filename_p2)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p2 Dataset Dim: ', df_noleak_rand_p2.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for no leak random -------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        noleak_data_p1 = df_noleak_rand_p1.loc[df_noleak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p1 = noleak_data_p1[np.random.permutation(len(noleak_data_p1))]

        noleak_data_p2 = df_noleak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p2 = noleak_data_p2[np.random.permutation(len(noleak_data_p2))]

        # merging all noleak scenario
        label = [0] * (len(noleak_data_p1) + len(noleak_data_p2))
        noleak_all = np.concatenate((noleak_data_p1, noleak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(noleak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed Noleak p1 dim: ', noleak_data_p1.shape)
        print('Preprocessed Noleak p2 dim: ', noleak_data_p2.shape)

        # for leak random ----------------------------------------------------------------------------------------------
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]

        leak_data_p2 = df_leak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p2 = leak_data_p2[np.random.permutation(len(leak_data_p2))]

        # merging all noleak scenario
        label = [1] * (len(leak_data_p1) + len(leak_data_p2))
        leak_all = np.concatenate((leak_data_p1, leak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed leak p1 dim: ', leak_data_p1.shape)
        print('Preprocessed leak p2 dim: ', leak_data_p2.shape)
        # concatenate all ----------------------------------------------------------------------------------------------
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

    def random_leak_noleak_downsampled(self, shuffle_b4_split=True, train_split=0.7):
        '''
        The leak of p1 contains 109 tdms files, and noleak of p1 only contains 49 tdms files. So we wil half the samples
        size from leak p1.
        The leak of p2 also contains 117 tdms files and noleak of p2 only containes 57 tdms files. to ensure balance
        class in training, half the sample size of leak p2.

        Also, remove the signals from channel -4m and -2m. we only want to train the model on recognizing leak no leak
        from 0m up to 10m, assuming the leak is always a one side of the sensor.
        '''
        # read leak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p1_ds)
        df_leak_rand_p1 = pd.read_csv(self.leak_dataset_filename_p1_ds)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p1 Dataset Dim: ', df_leak_rand_p1.values.shape)

        # read leak @ 0m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p2_ds)
        df_leak_rand_p2 = pd.read_csv(self.leak_dataset_filename_p2_ds)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p2 Dataset Dim: ', df_leak_rand_p2.values.shape)

        # read noleak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p1_ds)
        df_noleak_rand_p1 = pd.read_csv(self.noleak_dataset_filename_p1_ds)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p1 Dataset Dim: ', df_noleak_rand_p1.values.shape)

        # read noleak @ 0m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p2_ds)
        df_noleak_rand_p2 = pd.read_csv(self.noleak_dataset_filename_p2_ds)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p2 Dataset Dim: ', df_noleak_rand_p2.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for no leak random -------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        noleak_data_p1 = df_noleak_rand_p1.loc[df_noleak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p1 = noleak_data_p1[np.random.permutation(len(noleak_data_p1))]

        noleak_data_p2 = df_noleak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p2 = noleak_data_p2[np.random.permutation(len(noleak_data_p2))]

        # merging all noleak scenario
        label = [0] * (len(noleak_data_p1) + len(noleak_data_p2))
        noleak_all = np.concatenate((noleak_data_p1, noleak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(noleak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed Noleak p1 dim: ', noleak_data_p1.shape)
        print('Preprocessed Noleak p2 dim: ', noleak_data_p2.shape)

        # for leak random ----------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
            leak_data_p1 = leak_data_p1[:(len(leak_data_p1)//2)]  # **to balance the class

        leak_data_p2 = df_leak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p2 = leak_data_p2[np.random.permutation(len(leak_data_p2))]
            leak_data_p2 = leak_data_p2[:(len(leak_data_p2)//2)]  # **to balance the class

        # merging all noleak scenario
        label = [1] * (len(leak_data_p1) + len(leak_data_p2))
        leak_all = np.concatenate((leak_data_p1, leak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed leak p1 dim: ', leak_data_p1.shape)
        print('Preprocessed leak p2 dim: ', leak_data_p2.shape)

        # concatenate all ----------------------------------------------------------------------------------------------
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

    def random_leak_noleak_downsampled_include_unseen(self, shuffle_b4_split=True, train_split=0.7):
        '''
        This is same as above except added in unseen dataset into test set.
        '''
        print('DATASET: 2019-1-3 random_leak_noleak_downsampled_include_unseen()\n')
        # read leak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p1_ds)
        df_leak_rand_p1 = pd.read_csv(self.leak_dataset_filename_p1_ds)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p1 Dataset Dim: ', df_leak_rand_p1.values.shape)

        # read leak @ 0m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p2_ds)
        df_leak_rand_p2 = pd.read_csv(self.leak_dataset_filename_p2_ds)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p2 Dataset Dim: ', df_leak_rand_p2.values.shape)

        # read noleak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p1_ds)
        df_noleak_rand_p1 = pd.read_csv(self.noleak_dataset_filename_p1_ds)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p1 Dataset Dim: ', df_noleak_rand_p1.values.shape)

        # read noleak @ 0m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p2_ds)
        df_noleak_rand_p2 = pd.read_csv(self.noleak_dataset_filename_p2_ds)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p2 Dataset Dim: ', df_noleak_rand_p2.values.shape)

        # read noleak @ -3,5,7,16,17m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p3_ds)
        df_noleak_rand_p3 = pd.read_csv(self.noleak_dataset_filename_p3_ds)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p3 Dataset Dim: ', df_noleak_rand_p3.values.shape)

        # read leak @ -3,5,7,16,17m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p3_ds)
        df_leak_rand_p3 = pd.read_csv(self.leak_dataset_filename_p3_ds)
        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p3 Dataset Dim: ', df_leak_rand_p3.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for no leak random -------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        noleak_data_p1 = df_noleak_rand_p1.loc[df_noleak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p1 = noleak_data_p1[np.random.permutation(len(noleak_data_p1))]

        noleak_data_p2 = df_noleak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p2 = noleak_data_p2[np.random.permutation(len(noleak_data_p2))]

        # merging all noleak scenario
        label = [0] * (len(noleak_data_p1) + len(noleak_data_p2))
        noleak_all = np.concatenate((noleak_data_p1, noleak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(noleak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed Noleak p1 dim: ', noleak_data_p1.shape)
        print('Preprocessed Noleak p2 dim: ', noleak_data_p2.shape)

        # for leak random ----------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
        leak_data_p1 = leak_data_p1[:(len(leak_data_p1)//2)]  # **to balance the class

        leak_data_p2 = df_leak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p2 = leak_data_p2[np.random.permutation(len(leak_data_p2))]
        leak_data_p2 = leak_data_p2[:(len(leak_data_p2)//2)]  # **to balance the class

        # merging all noleak scenario
        label = [1] * (len(leak_data_p1) + len(leak_data_p2))
        leak_all = np.concatenate((leak_data_p1, leak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed leak p1 dim: ', leak_data_p1.shape)
        print('Preprocessed leak p2 dim: ', leak_data_p2.shape)

        # for unseen leak noleak on val data ---------------------------------------------------------------------------
        # noleak
        noleak_data_p3 = df_noleak_rand_p3.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p3 = noleak_data_p3[np.random.permutation(len(noleak_data_p3))]

        # leak
        leak_data_p3 = df_leak_rand_p3.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p3 = leak_data_p3[np.random.permutation(len(leak_data_p3))]
        leak_data_p3 = leak_data_p3[:(len(leak_data_p3) // 2)]  # **to balance the class, leak sample is twice noleak

        # merging all noleak scenario
        label = [0] * len(noleak_data_p3) + [1] * len(leak_data_p3)
        all = np.concatenate((noleak_data_p3, leak_data_p3), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        test_x.append(te_x)
        test_y.append(te_y)


        # # noleak
        # noleak_data_p3 = df_noleak_rand_p3.values[:, :-1]
        # if shuffle_b4_split:
        #     noleak_data_p3 = noleak_data_p3[np.random.permutation(len(noleak_data_p3))]
        #
        # label = [0] * len(noleak_data_p3)
        #
        # test_x.append(noleak_data_p3)
        # test_y.append(label)
        #
        # # leak
        # leak_data_p3 = df_leak_rand_p3.values[:, :-1]
        # if shuffle_b4_split:
        #     leak_data_p3 = leak_data_p3[np.random.permutation(len(leak_data_p3))]
        # leak_data_p3 = leak_data_p3[:(len(leak_data_p3) // 2)]  # **to balance the class
        #
        # label = [1] * len(leak_data_p3)

        # test_x.append(leak_data_p3)
        # test_y.append(label)

        print('Preprocessed noleak p3 dim: ', noleak_data_p3.shape)
        print('Preprocessed leak p3 dim: ', leak_data_p3.shape)

        # concatenate all ----------------------------------------------------------------------------------------------
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

    def random_leak_noleak_downsampled_2_include_unseen(self, shuffle_b4_split=True, train_split=0.7):
        '''
        DOWNSAMPLE FACTOR: 10
        SAMPLING RATE: 100kHz

        The leak of p1 contains 109 tdms files, and noleak of p1 only contains 49 tdms files. So we wil half the samples
        size from leak p1.
        The leak of p2 also contains 117 tdms files and noleak of p2 only containes 57 tdms files. to ensure balance
        class in training, half the sample size of leak p2.

        Also, remove the signals from channel -4m and -2m. we only want to train the model on recognizing leak no leak
        from 0m up to 10m, assuming the leak is always a one side of the sensor.
        '''
        print('DATASET: 2019-1-3 random_leak_noleak_downsampled_2_include_unseen()\n')
        # read leak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p1_ds2)
        df_leak_rand_p1 = pd.read_csv(self.leak_dataset_filename_p1_ds2)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p1 Dataset Dim: ', df_leak_rand_p1.values.shape)

        # read leak @ 0m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p2_ds2)
        df_leak_rand_p2 = pd.read_csv(self.leak_dataset_filename_p2_ds2)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p2 Dataset Dim: ', df_leak_rand_p2.values.shape)

        # read noleak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p1_ds2)
        df_noleak_rand_p1 = pd.read_csv(self.noleak_dataset_filename_p1_ds2)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p1 Dataset Dim: ', df_noleak_rand_p1.values.shape)

        # read noleak @ 0m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p2_ds2)
        df_noleak_rand_p2 = pd.read_csv(self.noleak_dataset_filename_p2_ds2)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p2 Dataset Dim: ', df_noleak_rand_p2.values.shape)

        # read noleak @ -3,5,7,16,17m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p3_ds2)
        df_noleak_rand_p3 = pd.read_csv(self.noleak_dataset_filename_p3_ds2)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p3 Dataset Dim: ', df_noleak_rand_p3.values.shape)

        # read leak @ -3,5,7,16,17m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p3_ds2)
        df_leak_rand_p3 = pd.read_csv(self.leak_dataset_filename_p3_ds2)
        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p3 Dataset Dim: ', df_leak_rand_p3.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for no leak random -------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        noleak_data_p1 = df_noleak_rand_p1.loc[df_noleak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p1 = noleak_data_p1[np.random.permutation(len(noleak_data_p1))]

        noleak_data_p2 = df_noleak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p2 = noleak_data_p2[np.random.permutation(len(noleak_data_p2))]

        # merging all noleak scenario
        label = [0] * (len(noleak_data_p1) + len(noleak_data_p2))
        noleak_all = np.concatenate((noleak_data_p1, noleak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(noleak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed Noleak p1 dim: ', noleak_data_p1.shape)
        print('Preprocessed Noleak p2 dim: ', noleak_data_p2.shape)

        # for leak random ----------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
        leak_data_p1 = leak_data_p1[:(len(leak_data_p1) // 2)]  # **to balance the class

        leak_data_p2 = df_leak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p2 = leak_data_p2[np.random.permutation(len(leak_data_p2))]
        leak_data_p2 = leak_data_p2[:(len(leak_data_p2) // 2)]  # **to balance the class

        # merging all noleak scenario
        label = [1] * (len(leak_data_p1) + len(leak_data_p2))
        leak_all = np.concatenate((leak_data_p1, leak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed leak p1 dim: ', leak_data_p1.shape)
        print('Preprocessed leak p2 dim: ', leak_data_p2.shape)

        # for unseen leak noleak on val data ---------------------------------------------------------------------------
        # noleak
        noleak_data_p3 = df_noleak_rand_p3.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p3 = noleak_data_p3[np.random.permutation(len(noleak_data_p3))]

        # leak
        leak_data_p3 = df_leak_rand_p3.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p3 = leak_data_p3[np.random.permutation(len(leak_data_p3))]
        leak_data_p3 = leak_data_p3[:(len(leak_data_p3) // 2)]  # **to balance the class, leak sample is twice noleak

        # merging all noleak scenario
        label = [0] * len(noleak_data_p3) + [1] * len(leak_data_p3)
        all = np.concatenate((noleak_data_p3, leak_data_p3), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        test_x.append(te_x)
        test_y.append(te_y)

        print('Preprocessed noleak p3 dim: ', noleak_data_p3.shape)
        print('Preprocessed leak p3 dim: ', leak_data_p3.shape)

        # concatenate all ----------------------------------------------------------------------------------------------
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

    def random_leak_noleak_downsampled_3_include_unseen(self, shuffle_b4_split=True, train_split=0.7):
        '''
        DOWNSAMPLE FACTOR: 20
        SAMPLING RATE: 50kHz

        '''
        print('DATASET: 2019-1-3 random_leak_noleak_downsampled_3_include_unseen()\n')

        # read leak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p1_ds3)
        df_leak_rand_p1 = pd.read_csv(self.leak_dataset_filename_p1_ds3)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p1 Dataset Dim: ', df_leak_rand_p1.values.shape)

        # read leak @ 0m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p2_ds3)
        df_leak_rand_p2 = pd.read_csv(self.leak_dataset_filename_p2_ds3)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p2 Dataset Dim: ', df_leak_rand_p2.values.shape)

        # read noleak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p1_ds3)
        df_noleak_rand_p1 = pd.read_csv(self.noleak_dataset_filename_p1_ds3)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p1 Dataset Dim: ', df_noleak_rand_p1.values.shape)

        # read noleak @ 0m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p2_ds3)
        df_noleak_rand_p2 = pd.read_csv(self.noleak_dataset_filename_p2_ds3)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p2 Dataset Dim: ', df_noleak_rand_p2.values.shape)

        # read noleak @ -3,5,7,16,17m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p3_ds3)
        df_noleak_rand_p3 = pd.read_csv(self.noleak_dataset_filename_p3_ds3)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p3 Dataset Dim: ', df_noleak_rand_p3.values.shape)

        # read leak @ -3,5,7,16,17m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p3_ds3)
        df_leak_rand_p3 = pd.read_csv(self.leak_dataset_filename_p3_ds3)
        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p3 Dataset Dim: ', df_leak_rand_p3.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for no leak random -------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        noleak_data_p1 = df_noleak_rand_p1.loc[df_noleak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p1 = noleak_data_p1[np.random.permutation(len(noleak_data_p1))]

        noleak_data_p2 = df_noleak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p2 = noleak_data_p2[np.random.permutation(len(noleak_data_p2))]

        # merging all noleak scenario
        label = [0] * (len(noleak_data_p1) + len(noleak_data_p2))
        noleak_all = np.concatenate((noleak_data_p1, noleak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(noleak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed Noleak p1 dim: ', noleak_data_p1.shape)
        print('Preprocessed Noleak p2 dim: ', noleak_data_p2.shape)

        # for leak random ----------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
        leak_data_p1 = leak_data_p1[:(len(leak_data_p1) // 2)]  # **to balance the class

        leak_data_p2 = df_leak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p2 = leak_data_p2[np.random.permutation(len(leak_data_p2))]
        leak_data_p2 = leak_data_p2[:(len(leak_data_p2) // 2)]  # **to balance the class

        # merging all noleak scenario
        label = [1] * (len(leak_data_p1) + len(leak_data_p2))
        leak_all = np.concatenate((leak_data_p1, leak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed leak p1 dim: ', leak_data_p1.shape)
        print('Preprocessed leak p2 dim: ', leak_data_p2.shape)

        # for unseen leak noleak on val data ---------------------------------------------------------------------------
        # noleak
        noleak_data_p3 = df_noleak_rand_p3.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p3 = noleak_data_p3[np.random.permutation(len(noleak_data_p3))]

        # leak
        leak_data_p3 = df_leak_rand_p3.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p3 = leak_data_p3[np.random.permutation(len(leak_data_p3))]
        leak_data_p3 = leak_data_p3[:(len(leak_data_p3) // 2)]  # **to balance the class, leak sample is twice noleak

        # merging all noleak scenario
        label = [0] * len(noleak_data_p3) + [1] * len(leak_data_p3)
        all = np.concatenate((noleak_data_p3, leak_data_p3), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        test_x.append(te_x)
        test_y.append(te_y)

        print('Preprocessed noleak p3 dim: ', noleak_data_p3.shape)
        print('Preprocessed leak p3 dim: ', leak_data_p3.shape)

        # concatenate all ----------------------------------------------------------------------------------------------
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

    def random_leak_noleak_downsampled_4_include_unseen(self, shuffle_b4_split=True, train_split=0.7):
        '''
        DOWNSAMPLE FACTOR: 40
        SAMPLING RATE: 25kHz
        '''
        print('DATASET: 2019-1-3 random_leak_noleak_downsampled_3_include_unseen()\n')

        # read leak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p1_ds4)
        df_leak_rand_p1 = pd.read_csv(self.leak_dataset_filename_p1_ds4)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p1 Dataset Dim: ', df_leak_rand_p1.values.shape)

        # read leak @ 0m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p2_ds4)
        df_leak_rand_p2 = pd.read_csv(self.leak_dataset_filename_p2_ds4)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p2 Dataset Dim: ', df_leak_rand_p2.values.shape)

        # read noleak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p1_ds4)
        df_noleak_rand_p1 = pd.read_csv(self.noleak_dataset_filename_p1_ds4)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p1 Dataset Dim: ', df_noleak_rand_p1.values.shape)

        # read noleak @ 0m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p2_ds4)
        df_noleak_rand_p2 = pd.read_csv(self.noleak_dataset_filename_p2_ds4)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p2 Dataset Dim: ', df_noleak_rand_p2.values.shape)

        # read noleak @ -3,5,7,16,17m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p3_ds4)
        df_noleak_rand_p3 = pd.read_csv(self.noleak_dataset_filename_p3_ds4)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p3 Dataset Dim: ', df_noleak_rand_p3.values.shape)

        # read leak @ -3,5,7,16,17m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p3_ds4)
        df_leak_rand_p3 = pd.read_csv(self.leak_dataset_filename_p3_ds4)
        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p3 Dataset Dim: ', df_leak_rand_p3.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for no leak random -------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        noleak_data_p1 = df_noleak_rand_p1.loc[df_noleak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p1 = noleak_data_p1[np.random.permutation(len(noleak_data_p1))]

        noleak_data_p2 = df_noleak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p2 = noleak_data_p2[np.random.permutation(len(noleak_data_p2))]

        # merging all noleak scenario
        label = [0] * (len(noleak_data_p1) + len(noleak_data_p2))
        noleak_all = np.concatenate((noleak_data_p1, noleak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(noleak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed Noleak p1 dim: ', noleak_data_p1.shape)
        print('Preprocessed Noleak p2 dim: ', noleak_data_p2.shape)

        # for leak random ----------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
        leak_data_p1 = leak_data_p1[:(len(leak_data_p1) // 2)]  # **to balance the class

        leak_data_p2 = df_leak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p2 = leak_data_p2[np.random.permutation(len(leak_data_p2))]
        leak_data_p2 = leak_data_p2[:(len(leak_data_p2) // 2)]  # **to balance the class

        # merging all noleak scenario
        label = [1] * (len(leak_data_p1) + len(leak_data_p2))
        leak_all = np.concatenate((leak_data_p1, leak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed leak p1 dim: ', leak_data_p1.shape)
        print('Preprocessed leak p2 dim: ', leak_data_p2.shape)

        # for unseen leak noleak on val data ---------------------------------------------------------------------------
        # noleak
        noleak_data_p3 = df_noleak_rand_p3.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p3 = noleak_data_p3[np.random.permutation(len(noleak_data_p3))]

        # leak
        leak_data_p3 = df_leak_rand_p3.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p3 = leak_data_p3[np.random.permutation(len(leak_data_p3))]
        leak_data_p3 = leak_data_p3[:(len(leak_data_p3) // 2)]  # **to balance the class, leak sample is twice noleak

        # merging all noleak scenario
        label = [0] * len(noleak_data_p3) + [1] * len(leak_data_p3)
        all = np.concatenate((noleak_data_p3, leak_data_p3), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        test_x.append(te_x)
        test_y.append(te_y)

        print('Preprocessed noleak p3 dim: ', noleak_data_p3.shape)
        print('Preprocessed leak p3 dim: ', leak_data_p3.shape)

        # concatenate all ----------------------------------------------------------------------------------------------
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

    def random_leak_noleak_downsampled_5_include_unseen(self, shuffle_b4_split=True, train_split=0.7):
        '''
        DOWNSAMPLE FACTOR: 500
        SAMPLING RATE: 2kHz

        '''
        print('DATASET: 2019-1-3 random_leak_noleak_downsampled_3_include_unseen()\n')

        # read leak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p1_ds5)
        df_leak_rand_p1 = pd.read_csv(self.leak_dataset_filename_p1_ds5)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p1 Dataset Dim: ', df_leak_rand_p1.values.shape)

        # read leak @ 0m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p2_ds5)
        df_leak_rand_p2 = pd.read_csv(self.leak_dataset_filename_p2_ds5)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p2 Dataset Dim: ', df_leak_rand_p2.values.shape)

        # read noleak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p1_ds5)
        df_noleak_rand_p1 = pd.read_csv(self.noleak_dataset_filename_p1_ds5)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p1 Dataset Dim: ', df_noleak_rand_p1.values.shape)

        # read noleak @ 0m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p2_ds5)
        df_noleak_rand_p2 = pd.read_csv(self.noleak_dataset_filename_p2_ds5)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p2 Dataset Dim: ', df_noleak_rand_p2.values.shape)

        # read noleak @ -3,5,7,16,17m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p3_ds5)
        df_noleak_rand_p3 = pd.read_csv(self.noleak_dataset_filename_p3_ds5)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p3 Dataset Dim: ', df_noleak_rand_p3.values.shape)

        # read leak @ -3,5,7,16,17m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p3_ds5)
        df_leak_rand_p3 = pd.read_csv(self.leak_dataset_filename_p3_ds5)
        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p3 Dataset Dim: ', df_leak_rand_p3.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for no leak random -------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        noleak_data_p1 = df_noleak_rand_p1.loc[df_noleak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p1 = noleak_data_p1[np.random.permutation(len(noleak_data_p1))]

        noleak_data_p2 = df_noleak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p2 = noleak_data_p2[np.random.permutation(len(noleak_data_p2))]

        # merging all noleak scenario
        label = [0] * (len(noleak_data_p1) + len(noleak_data_p2))
        noleak_all = np.concatenate((noleak_data_p1, noleak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(noleak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed Noleak p1 dim: ', noleak_data_p1.shape)
        print('Preprocessed Noleak p2 dim: ', noleak_data_p2.shape)

        # for leak random ----------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
        leak_data_p1 = leak_data_p1[:(len(leak_data_p1) // 2)]  # **to balance the class

        leak_data_p2 = df_leak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p2 = leak_data_p2[np.random.permutation(len(leak_data_p2))]
        leak_data_p2 = leak_data_p2[:(len(leak_data_p2) // 2)]  # **to balance the class

        # merging all noleak scenario
        label = [1] * (len(leak_data_p1) + len(leak_data_p2))
        leak_all = np.concatenate((leak_data_p1, leak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed leak p1 dim: ', leak_data_p1.shape)
        print('Preprocessed leak p2 dim: ', leak_data_p2.shape)

        # for unseen leak noleak on val data ---------------------------------------------------------------------------
        # noleak
        noleak_data_p3 = df_noleak_rand_p3.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p3 = noleak_data_p3[np.random.permutation(len(noleak_data_p3))]

        # leak
        leak_data_p3 = df_leak_rand_p3.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p3 = leak_data_p3[np.random.permutation(len(leak_data_p3))]
        leak_data_p3 = leak_data_p3[:(len(leak_data_p3) // 2)]  # **to balance the class, leak sample is twice noleak

        # merging all noleak scenario
        label = [0] * len(noleak_data_p3) + [1] * len(leak_data_p3)
        all = np.concatenate((noleak_data_p3, leak_data_p3), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        test_x.append(te_x)
        test_y.append(te_y)

        print('Preprocessed noleak p3 dim: ', noleak_data_p3.shape)
        print('Preprocessed leak p3 dim: ', leak_data_p3.shape)

        # concatenate all ----------------------------------------------------------------------------------------------
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

    def random_leak_noleak_downsampled_6_include_unseen(self, shuffle_b4_split=True, train_split=0.7):
        '''
        DOWNSAMPLE FACTOR: 1000
        SAMPLING RATE: 1kHz

        '''
        print('DATASET: 2019-1-3 random_leak_noleak_downsampled_3_include_unseen()\n')

        # read leak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p1_ds6)
        df_leak_rand_p1 = pd.read_csv(self.leak_dataset_filename_p1_ds6)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p1 Dataset Dim: ', df_leak_rand_p1.values.shape)

        # read leak @ 0m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p2_ds6)
        df_leak_rand_p2 = pd.read_csv(self.leak_dataset_filename_p2_ds6)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p2 Dataset Dim: ', df_leak_rand_p2.values.shape)

        # read noleak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p1_ds6)
        df_noleak_rand_p1 = pd.read_csv(self.noleak_dataset_filename_p1_ds6)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p1 Dataset Dim: ', df_noleak_rand_p1.values.shape)

        # read noleak @ 0m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p2_ds6)
        df_noleak_rand_p2 = pd.read_csv(self.noleak_dataset_filename_p2_ds6)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p2 Dataset Dim: ', df_noleak_rand_p2.values.shape)

        # read noleak @ -3,5,7,16,17m
        time_start = time.time()
        print('Reading --> ', self.noleak_dataset_filename_p3_ds6)
        df_noleak_rand_p3 = pd.read_csv(self.noleak_dataset_filename_p3_ds6)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p3 Dataset Dim: ', df_noleak_rand_p3.values.shape)

        # read leak @ -3,5,7,16,17m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p3_ds6)
        df_leak_rand_p3 = pd.read_csv(self.leak_dataset_filename_p3_ds6)
        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random NoLeak p3 Dataset Dim: ', df_leak_rand_p3.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for no leak random -------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        noleak_data_p1 = df_noleak_rand_p1.loc[df_noleak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p1 = noleak_data_p1[np.random.permutation(len(noleak_data_p1))]

        noleak_data_p2 = df_noleak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p2 = noleak_data_p2[np.random.permutation(len(noleak_data_p2))]

        # merging all noleak scenario
        label = [0] * (len(noleak_data_p1) + len(noleak_data_p2))
        noleak_all = np.concatenate((noleak_data_p1, noleak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(noleak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed Noleak p1 dim: ', noleak_data_p1.shape)
        print('Preprocessed Noleak p2 dim: ', noleak_data_p2.shape)

        # for leak random ----------------------------------------------------------------------------------------------
        # select only 2m, 4m, 6m, 8m, 10m
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
        leak_data_p1 = leak_data_p1[:(len(leak_data_p1) // 2)]  # **to balance the class

        leak_data_p2 = df_leak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p2 = leak_data_p2[np.random.permutation(len(leak_data_p2))]
        leak_data_p2 = leak_data_p2[:(len(leak_data_p2) // 2)]  # **to balance the class

        # merging all noleak scenario
        label = [1] * (len(leak_data_p1) + len(leak_data_p2))
        leak_all = np.concatenate((leak_data_p1, leak_data_p2), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        print('Preprocessed leak p1 dim: ', leak_data_p1.shape)
        print('Preprocessed leak p2 dim: ', leak_data_p2.shape)

        # for unseen leak noleak on val data ---------------------------------------------------------------------------
        # noleak
        noleak_data_p3 = df_noleak_rand_p3.values[:, :-1]
        if shuffle_b4_split:
            noleak_data_p3 = noleak_data_p3[np.random.permutation(len(noleak_data_p3))]

        # leak
        leak_data_p3 = df_leak_rand_p3.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p3 = leak_data_p3[np.random.permutation(len(leak_data_p3))]
        leak_data_p3 = leak_data_p3[:(len(leak_data_p3) // 2)]  # **to balance the class, leak sample is twice noleak

        # merging all noleak scenario
        label = [0] * len(noleak_data_p3) + [1] * len(leak_data_p3)
        all = np.concatenate((noleak_data_p3, leak_data_p3), axis=0)

        tr_x, te_x, tr_y, te_y = train_test_split(all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        test_x.append(te_x)
        test_y.append(te_y)

        print('Preprocessed noleak p3 dim: ', noleak_data_p3.shape)
        print('Preprocessed leak p3 dim: ', leak_data_p3.shape)

        # concatenate all ----------------------------------------------------------------------------------------------
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

    def random_leak_bydist_downsampled_4(self, shuffle_b4_split=True, train_split=0.8):
        '''
                DOWNSAMPLE FACTOR: 40
                SAMPLING RATE: 25kHz
        '''

        # read leak @ -4, -2, 2, 4, 6, 8, 10m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p1_ds4)
        df_leak_rand_p1 = pd.read_csv(self.leak_dataset_filename_p1_ds4)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p1 Dataset Dim: ', df_leak_rand_p1.values.shape)

        # read leak @ 0m
        time_start = time.time()
        print('Reading --> ', self.leak_dataset_filename_p2_ds4)
        df_leak_rand_p2 = pd.read_csv(self.leak_dataset_filename_p2_ds4)

        print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        print('Random Leak p2 Dataset Dim: ', df_leak_rand_p2.values.shape)

        # # read leak @ -3,5,7,16,17m
        # time_start = time.time()
        # print('Reading --> ', self.leak_dataset_filename_p3_ds4)
        # df_leak_rand_p3 = pd.read_csv(self.leak_dataset_filename_p3_ds4)
        # print('File Read Time: {:.4f}s'.format(time.time() - time_start))
        # print('Random NoLeak p3 Dataset Dim: ', df_leak_rand_p3.values.shape)

        train_x, train_y, test_x, test_y = [], [], [], []

        # for leak [0m] ----------------------------------------------------------------------------------------------
        leak_data_p2 = df_leak_rand_p2.values[:, :-1]
        if shuffle_b4_split:
            leak_data_p2 = leak_data_p2[np.random.permutation(len(leak_data_p2))]
        # leak_data_p2 = leak_data_p2[:(len(leak_data_p2) // 2)]  # **to reduce the class sample

        leak_all = leak_data_p2
        label = [0] * len(leak_data_p2)

        print('Leak @ 0m: {} samples'.format(len(leak_all)))

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for leak [2m] ----------------------------------------------------------------------------------------------
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'] == 2].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
        # leak_data_p1 = leak_data_p1[:(len(leak_data_p1) // 2)]  # **to balance the class

        leak_all = leak_data_p1
        label = [1] * len(leak_data_p1)

        print('Leak @ 2m: {} samples'.format(len(leak_all)))

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for leak [4m] ----------------------------------------------------------------------------------------------
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'] == 3].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
        # leak_data_p1 = leak_data_p1[:(len(leak_data_p1) // 2)]  # **to balance the class

        leak_all = leak_data_p1
        label = [2] * len(leak_data_p1)

        print('Leak @ 4m: {} samples'.format(len(leak_all)))

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for leak [6m] ----------------------------------------------------------------------------------------------
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'] == 4].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
        # leak_data_p1 = leak_data_p1[:(len(leak_data_p1) // 2)]  # **to balance the class

        leak_all = leak_data_p1
        label = [3] * len(leak_data_p1)

        print('Leak @ 6m: {} samples'.format(len(leak_all)))

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for leak [8m] ----------------------------------------------------------------------------------------------
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'] == 5].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
        # leak_data_p1 = leak_data_p1[:(len(leak_data_p1) // 2)]  # **to balance the class

        leak_all = leak_data_p1
        label = [4] * len(leak_data_p1)

        print('Leak @ 8m: {} samples'.format(len(leak_all)))

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # for leak [10m] ----------------------------------------------------------------------------------------------
        leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'] == 6].values[:, :-1]
        if shuffle_b4_split:
            leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
        # leak_data_p1 = leak_data_p1[:(len(leak_data_p1) // 2)]  # **to balance the class

        leak_all = leak_data_p1
        label = [5] * len(leak_data_p1)

        print('Leak @ 10m: {} samples'.format(len(leak_all)))

        tr_x, te_x, tr_y, te_y = train_test_split(leak_all,
                                                  label,
                                                  train_size=train_split,
                                                  shuffle=True)

        train_x.append(tr_x)
        test_x.append(te_x)
        train_y.append(tr_y)
        test_y.append(te_y)

        # concatenate all ----------------------------------------------------------------------------------------------
        train_x = np.concatenate(train_x, axis=0)
        test_x = np.concatenate(test_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        test_y = np.concatenate(test_y, axis=0)

        print('\n----------TRAIN AND TEST SET---------')
        print('Train_x Dim: ', train_x.shape)
        print('Val_x Dim: ', test_x.shape)
        print('Train_y Dim:', train_y.shape)
        print('Val_y Dim:', test_y.shape)

        return train_x, train_y, test_x, test_y


data = AcousticEmissionDataSet(drive='G')
data2 = data.random_leak_bydist_downsampled_4()




