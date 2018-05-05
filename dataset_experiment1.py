# ------------------------------------------------------
# Grab all data set from Harddisk, downsample and return
# as np matrix. Make sure Harddisk is E: drive
# ------------------------------------------------------

import numpy as np
from pandas import read_csv
from os import listdir
from scipy.signal import decimate
from utils import ProgressBarForLoop


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
    pb = ProgressBarForLoop('Reading CSV <-- ' + path, end=len(all_file_path))
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
        self.drive = 'F://'
        self.path_noleak_2bar_set1 = self.drive + 'Experiment 1//pos_0m_2m//No_Leak//2_bar//Set_1//'
        self.path_noleak_2bar_set2 = self.drive + 'Experiment 1//pos_0m_2m//No_Leak//2_bar//Set_2//'
        self.path_noleak_2bar_set3 = self.drive + 'Experiment 1//pos_0m_2m//No_Leak//2_bar//Set_3//'

    def noleak_2bar(self, sensor=1):
        '''
        :param sensor: Define the sensor no.
        :return: 2d np-array where rows = no of csv, col = len of csv
        Source Folder = 'Experiment 1//pos_0m_2m//No_Leak//2_bar//[ALL SET]//Sensor_N//'
        '''
        # define sensor number
        if sensor == 1:
            sensor_now = 'Sensor_1//'
        elif sensor == 2:
            sensor_now = 'Sensor_2//'

        # Complete Path
        path_now = [self.path_noleak_2bar_set1, self.path_noleak_2bar_set2, self.path_noleak_2bar_set3]
        for i in range(len(path_now)):
            path_now[i] += sensor_now

        # read csv from all folder
        data_temp = np.concatenate((read_csv_from_folder(path_now[0]),
                                    read_csv_from_folder(path_now[1]),
                                    read_csv_from_folder(path_now[2])),
                                   axis=0)
        print('Total Data Dim: {}'.format(data_temp.shape))
        return data_temp, data_temp.shape

    def testing(self):
        data_temp = read_csv_from_folder('dataset//')
        return data_temp










