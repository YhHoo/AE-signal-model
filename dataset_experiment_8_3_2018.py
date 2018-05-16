# ------------------------------------------------------
# DATASET: (Experiment_8_3_2018) --> in csv, each are 0.2 seconds only
# Grab all data set from Harddisk, downsample and return
# as np matrix. Make sure change the harddisk alphabet
# WARNING: Do not change the code here for TDMS file, create new file
# ------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from os import listdir
from scipy.signal import decimate
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

# self library
from utils import ProgressBarForLoop
from dsp_tools import spectrogram_scipy


def read_csv_from_folder(path=None):
    '''
    :param path: The folder that contains all 1d csv of same sizes
    :return: np-array where rows = no of csv, col = len of csv
    '''
    # ensure path is specified
    assert path is not None, 'No Folder is selected for CSV Extraction'
    # listdir(path) will return a list of file in location specified by path
    all_file_path = [(path + f) for f in listdir(path)]
    bank = []
    # call progressbar instance
    pb = ProgressBarForLoop('\nReading CSV <-- ' + path, end=len(all_file_path))
    for f in all_file_path:
        dataset = read_csv(f, skiprows=12, names=['Data_Point', 'Vibration_In_Volt'])
        # down-sampling from 5MHz to 1MHz
        dataset = decimate(dataset['Vibration_In_Volt'], q=5, zero_phase=True)
        bank.append(dataset)
        # update progress
        pb.update(all_file_path.index(f))
    # destroy progress bar
    pb.destroy()
    # convert the list to np array
    bank = np.array(bank)
    bank.reshape((len(bank), -1))
    print('Data Dim: {}'.format(bank.shape))

    return bank


# ----------------------[SPECIFY FILE HERE]-------------------------
# data files location options


class AcousticEmissionDataSet:
    def __init__(self):
        print('----------RAW DATA SET IMPORT & DOWN-SAMPLE---------')
        self.drive = 'E://'
        self.path_noleak_2bar_set1 = self.drive + 'Experiment_8_3_2018//pos_0m_2m//No_Leak//2_bar//Set_1//'
        self.path_noleak_2bar_set2 = self.drive + 'Experiment_8_3_2018//pos_0m_2m//No_Leak//2_bar//Set_2//'
        self.path_noleak_2bar_set3 = self.drive + 'Experiment_8_3_2018//pos_0m_2m//No_Leak//2_bar//Set_3//'
        self.path_leak_2bar_set1 = self.drive + 'Experiment_8_3_2018//pos_0m_2m//Leak//2_bar//Set_1//'
        self.path_leak_2bar_set2 = self.drive + 'Experiment_8_3_2018//pos_0m_2m//Leak//2_bar//Set_2//'
        self.path_leak_2bar_set3 = self.drive + 'Experiment_8_3_2018//pos_0m_2m//Leak//2_bar//Set_3//'

    def noleak_2bar(self, sensor=1):
        '''
        :param sensor: Define the sensor no.
        :return: 2d np-array where rows = no of csv, col = len of csv
        Source Folder = 'Experiment_8_3_2018//pos_0m_2m//No_Leak//2_bar//[ALL SET]//Sensor_N//'
        '''
        # define sensor number
        if sensor == 1:
            sensor_now = 'Sensor_1//'
        elif sensor == 2:
            sensor_now = 'Sensor_2//'

        # Complete Path with sensor no
        path_now = [self.path_noleak_2bar_set1, self.path_noleak_2bar_set2, self.path_noleak_2bar_set3]
        for i in range(len(path_now)):
            path_now[i] += sensor_now

        # read csv from all folder
        data_temp = np.concatenate((read_csv_from_folder(path_now[0]),
                                    read_csv_from_folder(path_now[1]),
                                    read_csv_from_folder(path_now[2])),
                                   axis=0)
        print('Total [noleak_2bar] Data Dim: {}'.format(data_temp.shape))
        return data_temp

    def leak_2bar(self, sensor=1):
        '''
        :param sensor: Define the sensor no.
        :return: 2d np-array where rows = no of csv, col = len of csv
        Source Folder = 'Experiment_8_3_2018//pos_0m_2m//No_Leak//2_bar//[ALL SET]//Sensor_N//'
        '''
        # define sensor number
        if sensor == 1:
            sensor_now = 'Sensor_1//'
        elif sensor == 2:
            sensor_now = 'Sensor_2//'

        # Complete Path with sensor no
        path_now = [self.path_leak_2bar_set1, self.path_leak_2bar_set2, self.path_leak_2bar_set3]
        for i in range(len(path_now)):
            path_now[i] += sensor_now

        # read csv from all folder
        data_temp = np.concatenate((read_csv_from_folder(path_now[0]),
                                    read_csv_from_folder(path_now[1]),
                                    read_csv_from_folder(path_now[2])),
                                   axis=0)
        print('Total [leak_2bar] Data Dim: {}'.format(data_temp.shape))
        return data_temp

    def testing(self):
        data_temp = read_csv_from_folder('dataset//')
        return data_temp


# ----------------------[DATA IMPORT]-------------------------
ae_dataset_1 = AcousticEmissionDataSet()

# testing
data_noleak = ae_dataset_1.noleak_2bar(sensor=1)
data_leak = ae_dataset_1.leak_2bar(sensor=1)

# ----------------------[SIGNAL TRANSFORMATION]-------------------------
print('\n----------SIGNAL TRANSFORMATION---------')
# for holding all fft 2d array
data_noleak_all, data_leak_all = [], []

# loop through all rows in data_noleak
pb = ProgressBarForLoop('Spectrogram Transform --> [noleak_2bar]', end=data_noleak.shape[0])
flag = 0
for signal_in_time in data_noleak[:]:
    _, _, mat = spectrogram_scipy(signal_in_time, fs=1e6, visualize=False, verbose=False)
    # take 0-300kHz only
    data_noleak_all.append(mat[:3000])
    pb.update(now=flag)
    flag += 1
pb.destroy()
data_noleak_all = np.array(data_noleak_all)
print('NoLeak Data Time Frequency Rep Dim: {}\n'.format(data_noleak_all.shape))

# loop through all rows in data_leak
pb = ProgressBarForLoop('Spectrogram Transform --> [leak_2bar]', end=data_leak.shape[0])
flag = 0
for signal_in_time in data_leak[:]:
    _, _, mat = spectrogram_scipy(signal_in_time, fs=1e6, visualize=False, verbose=False)
    # take 0-300kHz only
    data_leak_all.append(mat[:3000])
    pb.update(now=flag)
    flag += 1
pb.destroy()
data_leak_all = np.array(data_leak_all)
print('\nLeak Data Time Frequency Rep Dim: {}\n'.format(data_leak_all.shape))

# ----------------------[TEST AND TRAIN DATA PREPARATION]-------------------------
# control the split
test_split = 0.7
split_index = int(0.7 * data_noleak_all.shape[0])

# slicing and concatenation
train_x = np.concatenate((data_noleak_all[:split_index], data_leak_all[:split_index]), axis=0)
train_y = np.array([0] * split_index + [1] * split_index)
test_x = np.concatenate((data_noleak_all[split_index:], data_leak_all[split_index:]), axis=0)
test_y = np.array([0] * (data_noleak_all.shape[0] - split_index) + [1] * (data_leak_all.shape[0] - split_index))

# reshape to satisfy conv2d input shape
train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))

# convert
train_y = to_categorical(train_y, num_classes=2)
test_y = to_categorical(test_y, num_classes=2)

# data summary
print('\n----------INPUT DATA DIMENSION---------')
print('Train_X dim: ', train_x.shape)
print('Train_Y dim: ', train_y.shape)
print('Test_X dim: ', test_x.shape)
print('Test_Y dim: ', test_y.shape)


# l = np.array([[[1, 2, 3], [4, 5, 6]]])
# m = np.array([[[11, 22, 33], [44, 55, 66]]])
# print(l)
# print(m)
#
# n = np.concatenate((l, m), axis=0)
# print(n)
# print(n.shape)

# ----------------------[MODEL TRAINING AND TESTING]-------------------------
model = Sequential()
# Convolutional layer 1 ------------------------------------------
model.add(Conv2D(filters=36, kernel_size=(10, 5), strides=(1, 1),
                 activation='relu', input_shape=(3000, 23, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

# Convolutional layer 2 ------------------------------------------
model.add(Conv2D(filters=72, kernel_size=(10, 5), strides=(2, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 2), strides=(2, 1)))

# Convolutional layer 3 ------------------------------------------
model.add(Conv2D(filters=96, kernel_size=(10, 3), strides=(4, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 1), strides=(3, 1)))

# Convolutional layer 4 ------------------------------------------
model.add(Conv2D(filters=109, kernel_size=(6, 1), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(9, 1), strides=(2, 1)))
model.add(Flatten())

# Fully connected layer 1 ----------------------------------------
model.add(Dense(150, activation='relu'))

# Fully connected layer 2 ----------------------------------------
model.add(Dense(80, activation='relu'))

# Fully connected layer 3 ----------------------------------------
model.add(Dense(2, activation='softmax'))
# print architecture summary
print(model.summary())

# start training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x=train_x,
                    y=train_y,
                    batch_size=10,
                    validation_data=(test_x, test_y),
                    verbose=2,
                    epochs=4)

# visualize of training process
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.legend()
plt.show()

