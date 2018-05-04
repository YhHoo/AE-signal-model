# ------------------------------------------------------
# Grab all data set from Harddisk, downsample and return
# as np matrix. Make sure Harddisk is E: drive
# ------------------------------------------------------

import numpy as np
from pandas import read_csv
from os import listdir
from os.path import isfile, join
from scipy.signal import decimate


# data files location
path_noleak_2bar_set1 = 'E://Experiment 1//pos_0m_2m//No_Leak//2_bar//Set_1//'
path_noleak_2bar_set2 = 'E://Experiment 1//pos_0m_2m//No_Leak//2_bar//Set_2//Sensor_1//'
path_noleak_2bar_set3 = 'E://Experiment 1//pos_0m_2m//No_Leak//2_bar//Set_3//Sensor_1//'
sensor1 = 'Sensor_1//'
sensor2 = 'Sensor_2//'

# listdir(path) will return a list of file in location specified by path
all_file_path = [(path_mendeley + f) for f in listdir(path_mendeley)]
    # list of all filename only
all_filename = [full_path[(len(path_mendeley)):] for full_path in all_file_path]


data_noleak_raw_1 = read_csv(path_noleak_2bar + data_1 + '.csv',
                             skiprows=12,
                             names=['Data_Point', 'Vibration_In_Volt'])