import numpy as np
import matplotlib.pyplot as plt
# self lib
from src.experiment_dataset.dataset_experiment_30_5_2018 import AcousticEmissionDataSet_30_5_2018
from src.utils.dsp_tools import one_dim_xcor_freq_band, spectrogram_scipy
from src.utils.helpers import three_dim_visualizer

# -------------------[Xcor testing of spectrogram output]-------------------
data = AcousticEmissionDataSet_30_5_2018(drive='E')
ft_mode = 'magnitude'  # mag, angle, phase

# data acquisition for leak pos @ 0m----------------
n_channel_data, _, _, _ = data.plb_4_sensor(leak_pos=0)
# plotting the time series data of the sensors
fig1 = plt.figure()
fig1.subplots_adjust(hspace=0.5)
# fig 1
ax11 = fig1.add_subplot(2, 1, 1)
ax11.set_title('Time series sensor [-1m] @ 0m')
ax12 = fig1.add_subplot(2, 1, 2)
ax12.set_title('Time series sensor [22m] @ 0m')
ax11.plot(n_channel_data[1, 500000:1500000, 1])
ax12.plot(n_channel_data[1, 500000:1500000, 2])

_, freq_axis, sxx1, figA1 = spectrogram_scipy(sampled_data=n_channel_data[1, 500000:1500000, 1],
                                              fs=1e6,
                                              nperseg=1000,
                                              noverlap=0,
                                              mode=ft_mode,
                                              return_plot=True,
                                              plot_title='Freq-Time rep of sensor [-1m] @ 0m',
                                              verbose=True, vis_max_freq_range=1e5)

_, _, sxx2, figB1 = spectrogram_scipy(sampled_data=n_channel_data[1, 500000:1500000, 2],
                                      fs=1e6,
                                      nperseg=1000,
                                      noverlap=0,
                                      mode=ft_mode,
                                      return_plot=True,
                                      plot_title='Freq-Time rep of sensor [22m] @ 0m',
                                      verbose=True, vis_max_freq_range=1e5)
mag_map = np.array([sxx1, sxx2])
sensor_pair = [(0, 1)]
xcor_map = one_dim_xcor_freq_band(input_mat=mag_map,
                                  pair_list=sensor_pair,
                                  verbose=True)
xcor_fig_1 = three_dim_visualizer(x_axis=np.arange(1, xcor_map.shape[2] + 1, 1),
                                  y_axis=freq_axis,
                                  zxx=xcor_map[0],
                                  label=['Xcor_steps', 'Frequency', 'Correlation Score'],
                                  output='2d',
                                  title='PLB Mag Map Xcor - Sensor[-1m] x Sensor[22m] - Set1 @ 0m')

# data acquisition for leak pos @ 2m----------------
n_channel_data, _, _, _ = data.plb_4_sensor(leak_pos=2)
# plotting the time series data of the sensors
fig1 = plt.figure()
fig1.subplots_adjust(hspace=0.5)
# fig 1
ax11 = fig1.add_subplot(2, 1, 1)
ax11.set_title('Time series sensor [-1m] @ 2m')
ax12 = fig1.add_subplot(2, 1, 2)
ax12.set_title('Time series sensor [22m] @ 2m')
ax11.plot(n_channel_data[1, 500000:1500000, 1])
ax12.plot(n_channel_data[1, 500000:1500000, 2])

_, freq_axis, sxx1, figA2 = spectrogram_scipy(sampled_data=n_channel_data[1, 500000:1500000, 1],
                                              fs=1e6,
                                              nperseg=1000,
                                              noverlap=0,
                                              mode=ft_mode,
                                              return_plot=True,
                                              plot_title='Freq-Time rep of sensor [-1m] @ 2m',
                                              verbose=True, vis_max_freq_range=1e5)

_, _, sxx2, figB2 = spectrogram_scipy(sampled_data=n_channel_data[1, 500000:1500000, 2],
                                      fs=1e6,
                                      nperseg=1000,
                                      noverlap=0,
                                      mode=ft_mode,
                                      return_plot=True,
                                      plot_title='Freq-Time rep of sensor [22m] @ 2m',
                                      verbose=True, vis_max_freq_range=1e5)
mag_map = np.array([sxx1, sxx2])
sensor_pair = [(0, 1)]
xcor_map = one_dim_xcor_freq_band(input_mat=mag_map,
                                  pair_list=sensor_pair,
                                  verbose=True)
xcor_fig_2 = three_dim_visualizer(x_axis=np.arange(1, xcor_map.shape[2] + 1, 1),
                                  y_axis=freq_axis,
                                  zxx=xcor_map[0],
                                  label=['Xcor_steps', 'Frequency', 'Correlation Score'],
                                  output='2d',
                                  title='PLB Mag Map Xcor - Sensor[-1m] x Sensor[22m] - Set1 @ 2m')

plt.show()

# ----------------------[Visualize in Time and Saving]----------------------------
time_analysis = False

if time_analysis:
    save_path = 'C:/Users/YH/Desktop/hooyuheng.masterWork/MASTER_PAPERWORK/' \
                'My Practical Work------------/Exp30_5_2018/PLB test/time series/leak @ 6m/'
    # for all sets of the same setup
    for set_no in range(n_channel_data.shape[0]):
        plt.subplot(4, 1, 1)
        plt.plot(n_channel_data[set_no, :, 0])
        plt.title('sensor @ -2m')
        plt.subplot(4, 1, 2)
        plt.plot(n_channel_data[set_no, :, 1])
        plt.title('sensor @ -1m')
        plt.subplot(4, 1, 3)
        plt.plot(n_channel_data[set_no, :, 2])
        plt.title('sensor @ 22m')
        plt.subplot(4, 1, 4)
        plt.plot(n_channel_data[set_no, :, 3])
        plt.title('sensor @ 23m')
        path = save_path + 'set_{}'.format(set_no)
        plt.savefig(path)
        plt.close()
        print('Saved !')

# ----------------------[Xcor for Phase Maps]----------------------------
xcor_analysis = False
save_path = 'C:/Users/YH/Desktop/hooyuheng.masterWork/MASTER_PAPERWORK/' \
            'My Practical Work------------/Exp30_5_2018/PLB test/'

if xcor_analysis:
    # xcor for sensor at -1m and 22m
    sensor_pair = [(0, 1), (1, 2), (0, 3), (2, 3)]
    label = [(-2, -1), (-1, 22), (-2, 23), (22, 23)]

    set_no = 0
    filename = 0
    # for all samples sets
    for set in phase_bank:
        xcor_map = one_dim_xcor_freq_band(input_mat=set,
                                          pair_list=sensor_pair,
                                          verbose=True)
        # max_xscore = []
        # # for all freq bands, take the pos where max xscore happens
        # for row in xcor_map[2]:
        #     max_xscore.append(np.argmax(row))
        # plt.plot(freq_axis, max_xscore, marker='x')
        # plt.show()

        j = 0
        for map in xcor_map:
            fig = three_dim_visualizer(x_axis=np.arange(1, map.shape[1] + 1, 1),
                                       y_axis=freq_axis,
                                       zxx=map,
                                       label=['Xcor_steps', 'Frequency', 'Correlation Score'],
                                       output='2d',
                                       title='PLB Phase Map - Sensor[{}m] x Sensor[{}m] - Set{}'
                                       .format(label[j][0], label[j][1], set_no))

            fig.savefig('C:/Users/YH/Desktop/hooyuheng.masterWork/MASTER_PAPERWORK/'
                        'My Practical Work------------/Exp30_5_2018/PLB test/{}.png'.format(filename))

            print('saved !')
            plt.close()
            filename += 1
            j += 1
        set_no += 1
