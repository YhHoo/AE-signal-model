import numpy as np
import matplotlib.pyplot as plt
# self lib
from src.experiment_dataset.dataset_experiment_30_5_2018 import AcousticEmissionDataSet_30_5_2018
from src.utils.dsp_tools import one_dim_xcor_freq_band
from src.utils.helpers import three_dim_visualizer


data = AcousticEmissionDataSet_30_5_2018(drive='E')
n_channel_data, phase_bank, freq_axis = data.plb_4_sensor(leak_pos=4)

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

