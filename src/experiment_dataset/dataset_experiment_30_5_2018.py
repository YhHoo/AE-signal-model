import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# self library
from src.utils.helpers import read_all_tdms_from_folder, three_dim_visualizer
from src.utils.dsp_tools import spectrogram_scipy


# for Pencil Lead Break data, the sensor position are (-2, -1, 22, 23)m
class AcousticEmissionDataSet_30_5_2018:
    def __init__(self, drive):
        self.drive = drive + ':/'
        self.path_0m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/0m/PLB/'
        self.path_2m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/2m/PLB/'
        self.path_4m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/4m/PLB/'
        self.path_6m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/6m/PLB/'

    def plb_4_sensor(self, leak_pos=0):
        if leak_pos is 0:
            n_channel_data = read_all_tdms_from_folder(self.path_0m_plb)
        elif leak_pos is 2:
            n_channel_data = read_all_tdms_from_folder(self.path_0m_plb)
        elif leak_pos is 4:
            n_channel_data = read_all_tdms_from_folder(self.path_0m_plb)
        elif leak_pos is 6:
            n_channel_data = read_all_tdms_from_folder(self.path_0m_plb)

        return n_channel_data


data = AcousticEmissionDataSet_30_5_2018(drive='E')
sensor_data = data.plb_4_sensor(leak_pos=0)

# plt.subplot(4, 1, 1)
# plt.plot(sensor_data[1, :, 0])
# plt.title('sensor @ -2m')
# plt.subplot(4, 1, 2)
# plt.plot(sensor_data[1, :, 1])
# plt.title('sensor @ -1m')
# plt.subplot(4, 1, 3)
# plt.plot(sensor_data[1, :, 2])
# plt.title('sensor @ 22m')
# plt.subplot(4, 1, 4)
# plt.plot(sensor_data[1, :, 3])
# plt.title('sensor @ 23m')
#
# plt.show()
# plt.close()

# take set 1 of leak pos 0m
phase_map = []
# for all 4 sensors
for i in range(sensor_data.shape[2]):
    t, f, Sxx = spectrogram_scipy(sensor_data[1, 800000:1000000, i],
                                  fs=1e6,
                                  nperseg=2000,
                                  noverlap=0,
                                  mode='angle',
                                  visualize=False,
                                  verbose=True,
                                  vis_max_freq_range=1e6/2)
    phase_map.append(Sxx)
phase_map = np.array(phase_map)

# xcor for sensor at -1m and 22m
# for all frequency bands
xcor_of_each_f_list = []
normalize_xcor_score = True
scaler = MinMaxScaler(feature_range=(0, 1))
sensor_pair = [(0, 1), (1, 2), (2, 3)]
label = [(-2, -1), (-1, 22), (22, 23)]

# for all sensor pair
for i in range(len(sensor_pair)):
    # for all frequency bands
    for k in range(phase_map.shape[1]):
        x_cor = np.correlate(phase_map[sensor_pair[i][0], k], phase_map[sensor_pair[i][1], k], 'full')
        xcor_of_each_f_list.append(x_cor)
    # xcor map of 2 phase map, axis[0] is freq, axis[1] is x-cor unit shift
    xcor_of_each_f_list = np.array(xcor_of_each_f_list)

    if normalize_xcor_score:
        # normalize each xcor_map with linear function btw their max and min values
        xcor_of_each_f_list = scaler.fit_transform(xcor_of_each_f_list.ravel().reshape((-1, 1)))\
            .reshape((xcor_of_each_f_list.shape[0], xcor_of_each_f_list.shape[1]))

    three_dim_visualizer(x_axis=np.arange(1, xcor_of_each_f_list.shape[1] + 1, 1),
                         y_axis=f,
                         zxx=xcor_of_each_f_list,
                         label=['Xcor_steps', 'Frequency', 'Correlation Score'],
                         output='color_map',
                         title='PLB Phase Map - Sensor[{}m] x Sensor[{}m]'.format(label[i][0], label[i][1]))
    xcor_of_each_f_list = []
