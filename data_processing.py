# ------------------------------------------------------
# Process the Raw AE signals for training-ready
# The Raw signal is sampled at 5MHz, So time btw points = 2e-7 s
# ------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv


# files location
path_noleak_2bar = 'E://Experiment 1//pos_0m_2m//No_Leak//2_bar//Set_1//'
path_leak_2bar = 'E://Experiment 1//pos_0m_2m//Leak//2_bar//Set_1//'
# the sensors
sensor_1 = 'STREAM 06.03.201820180306-143237-780_1_1048500_2096999'
sensor_2 = 'STREAM 06.03.201820180306-143732-581_1_1048500_2096999'

data_noleak_raw = read_csv(path_noleak_2bar + sensor_1 + '.csv',
                           skiprows=12,
                           names=['Data_Point', 'Vibration_In_Volt'])
print(data_noleak_raw.head())
print(data_noleak_raw.shape)

plt.plot(data_noleak_raw['Vibration_In_Volt'])
plt.show()
