import numpy as np
import matplotlib.pyplot as plt
# self library
from src.utils.helpers import read_all_tdms_from_folder, three_dim_visualizer, ProgressBarForLoop
from src.utils.dsp_tools import spectrogram_scipy, butter_bandpass_filtfilt, one_dim_xcor_2d_input


class AcousticEmissionDataSet_13_7_2018:
    '''
    The sensor position are (-3, -2, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)m
    '''
    def __init__(self, drive):
        self.drive = drive + ':/'
        self.path_plb_2to12 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/PLB/0m/'
        self.path_plb_10to22 = self.drive + 'Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/PLB/0m/'

    def plb(self):
        '''
        This function returns a dataset of xcor map, each belongs to a class of PLB captured by 2 sensors at different
        distance.
        :return:
        Xcor map
        '''
        n_channel_data = read_all_tdms_from_folder(self.path_plb_1bar)
        fig = plt.figure()


data = AcousticEmissionDataSet_13_7_2018(drive='F')
data.plb()