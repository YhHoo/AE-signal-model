import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# self library
from src.utils.helpers import read_all_tdms_from_folder, three_dim_visualizer
from src.utils.dsp_tools import spectrogram_scipy, one_dim_xcor_freq_band


# for Pencil Lead Break data, the sensor position are (-2, -1, 22, 23)m
class AcousticEmissionDataSet_30_5_2018:
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
            n_channel_data = read_all_tdms_from_folder(self.path_0m_plb)
        elif leak_pos is 4:
            n_channel_data = read_all_tdms_from_folder(self.path_0m_plb)
        elif leak_pos is 6:
            n_channel_data = read_all_tdms_from_folder(self.path_0m_plb)

        # save_path = 'C:/Users/YH/Desktop/hooyuheng.masterWork/MASTER_PAPERWORK/' \
        #             'My Practical Work------------/Exp30_5_2018/PLB test/'
        # for set_no in range(n_channel_data.shape[0]):
        #     plt.subplot(4, 1, 1)
        #     plt.plot(n_channel_data[set_no, :, 0])
        #     plt.title('sensor @ -2m')
        #     plt.subplot(4, 1, 2)
        #     plt.plot(n_channel_data[set_no, :, 1])
        #     plt.title('sensor @ -1m')
        #     plt.subplot(4, 1, 3)
        #     plt.plot(n_channel_data[set_no, :, 2])
        #     plt.title('sensor @ 22m')
        #     plt.subplot(4, 1, 4)
        #     plt.plot(n_channel_data[set_no, :, 3])
        #     plt.title('sensor @ 23m')
        #     path = save_path + 'set_{}'.format(set_no)
        #     plt.savefig(path)
        #     plt.close()
        #     print('Saved !')

        # ---------------------[STFT into phase maps]------------------------

        # for all sets (samples)
        phase_map_all = []
        for set_no in range(n_channel_data.shape[0]):
            phase_map_bank = []
            # for all sensors
            for sensor_no in range(n_channel_data.shape[2]):
                t, f, Sxx = spectrogram_scipy(n_channel_data[set_no, 800000:1000000, sensor_no],
                                              fs=1e6,
                                              nperseg=2000,
                                              noverlap=0,
                                              mode='angle',
                                              visualize=False,
                                              verbose=False,
                                              vis_max_freq_range=1e6 / 2)
                phase_map_bank.append(Sxx)
            phase_map_bank = np.array(phase_map_bank)
            phase_map_all.append(phase_map_bank)
        # convert to array
        phase_map_all = np.array(phase_map_all)
        print('Phase Map Dim (set_no, sensor_no, freq_band, time steps): ', phase_map_all.shape)

        return phase_map_all, f


leak_pos = [0, 2, 4, 6]


data = AcousticEmissionDataSet_30_5_2018(drive='E')
phase_bank, freq_axis = data.plb_4_sensor(leak_pos=0)

# xcor for sensor at -1m and 22m
sensor_pair = [(0, 1), (1, 2), (2, 3)]
label = [(-2, -1), (-1, 22), (22, 23)]
sample_no = 1

xcor_map = one_dim_xcor_freq_band(input_mat=phase_bank[sample_no],
                                  pair_list=sensor_pair,
                                  verbose=True)
max_xscore = []
# for all freq bands, take the pos where max xscore happens
for row in xcor_map[2]:
    max_xscore.append(np.argmax(row))
plt.plot(freq_axis, max_xscore, marker='x')
plt.show()

# i = 0
# for map in xcor_map:
#     three_dim_visualizer(x_axis=np.arange(1, map.shape[1] + 1, 1),
#                          y_axis=freq_axis,
#                          zxx=map,
#                          label=['Xcor_steps', 'Frequency', 'Correlation Score'],
#                          output='2d',
#                          title='PLB Phase Map - Sensor[{}m] x Sensor[{}m]'.format(label[i][0], label[i][1]))
#     i += 1
