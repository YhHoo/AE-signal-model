# ------------------------------------------------------
# Grab all data set from Harddisk, downsample and return
# as np matrix. Make sure Harddisk is E: drive
# ------------------------------------------------------

import numpy as np
from pandas import read_csv
from os import listdir
from os.path import isfile, join
from scipy.signal import decimate
from utils import ProgressBarForLoop


# data files location
drive = 'F://'
path_noleak_2bar_set1 = drive + 'Experiment 1//pos_0m_2m//No_Leak//2_bar//Set_1//'
path_noleak_2bar_set2 = drive + 'Experiment 1//pos_0m_2m//No_Leak//2_bar//Set_2//'
path_noleak_2bar_set3 = drive + 'Experiment 1//pos_0m_2m//No_Leak//2_bar//Set_3//'
sensor1 = 'Sensor_1//'
sensor2 = 'Sensor_2//'

# listdir(path) will return a list of file in location specified by path
all_file_path = [(path_noleak_2bar_set1 + sensor1 + f) for f in listdir(path_noleak_2bar_set1 + sensor1)]

pb = ProgressBarForLoop('Reading CSV from', end=len(all_file_path))
for f in all_file_path:
    dataset = read_csv(f, skiprows=12, names=['Data_Point', 'Vibration_In_Volt'])
    pb.update(all_file_path.index(f))

pb.destroy()


