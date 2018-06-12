import numpy as np
# self library
from src.utils.helpers import read_all_tdms_from_folder
from src.utils.dsp_tools import spectrogram_scipy


class AcousticEmissionDataSet_30_5_2018:
    '''
    The sensor position are (-2, -1, 22, 23)m
    '''
    def __init__(self, drive):
        self.drive = drive + ':/'
        self.path_0m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/0m/PLB/'
        self.path_2m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/2m/PLB/'
        self.path_4m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/4m/PLB/'
        self.path_6m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/6m/PLB/'

    def plb_4_sensor(self, leak_pos=0):
        # ---------------------[Select the file and read]------------------------
        '''
        n_channel_data is a 3d matrix where shape[0]-> no of set(sample size),
                                            shape[1]-> no. of AE data points,
                                            shape[2]-> no. of sensors
        '''
        if leak_pos is 0:
            n_channel_data = read_all_tdms_from_folder(self.path_0m_plb)
        elif leak_pos is 2:
            n_channel_data = read_all_tdms_from_folder(self.path_2m_plb)
        elif leak_pos is 4:
            n_channel_data = read_all_tdms_from_folder(self.path_4m_plb)
        elif leak_pos is 6:
            n_channel_data = read_all_tdms_from_folder(self.path_6m_plb)

        # ---------------------[STFT into phase maps]------------------------

        # for all sets (samples)
        phase_map_all = []
        for set_no in range(n_channel_data.shape[0]):
            phase_map_bank = []
            # for all sensors
            for sensor_no in range(n_channel_data.shape[2]):
                t, f, Sxx = spectrogram_scipy(n_channel_data[set_no, 500000:1500000, sensor_no],
                                              fs=1e6,
                                              nperseg=2000,
                                              noverlap=0,
                                              mode='angle',
                                              visualize=False,
                                              verbose=True,
                                              vis_max_freq_range=1e6 / 2)
                phase_map_bank.append(Sxx)
            phase_map_bank = np.array(phase_map_bank)
            phase_map_all.append(phase_map_bank)
        # convert to array
        phase_map_all = np.array(phase_map_all)
        print('Phase Map Dim (set_no, sensor_no, freq_band, time steps): ', phase_map_all.shape, '\n')

        return n_channel_data, phase_map_all, f


