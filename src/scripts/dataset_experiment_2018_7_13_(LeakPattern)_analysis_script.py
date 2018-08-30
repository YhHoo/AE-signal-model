import matplotlib.pyplot as plt
import pywt
import numpy as np
# self lib
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.helpers import direct_to_dir, read_all_tdms_from_folder, plot_cwt_with_time_series
from src.utils.dsp_tools import one_dim_xcor_2d_input


# CONFIG --------------------------------------------------------------------------------------------------------------
# wavelet
m_wavelet = 'gaus1'
scale = np.linspace(2, 30, 100)
fs = 1e6

# segmentation
no_of_segment = 1

# DATA POINT ----------------------------------------------------------------------------------------------------------
# read leak data
on_pc = True
if on_pc:
    data = AcousticEmissionDataSet_13_7_2018(drive='F')
    n_channel_leak = data.test_data(sensor_dist='near', leak='plb')
else:
    data_dir = direct_to_dir(where='yh_laptop_test_data') + 'plb/'
    n_channel_leak = read_all_tdms_from_folder(data_dir)
    n_channel_leak = np.swapaxes(n_channel_leak, 1, 2)
    n_channel_leak = n_channel_leak[0]

# processing
print(n_channel_leak.shape)

# break into a list of segmented points
n_channel_leak = np.split(n_channel_leak, axis=1, indices_or_sections=no_of_segment)
print('Total Segment: ', len(n_channel_leak))
print('Each Segment Dim: ', n_channel_leak[0].shape)


# CWT + XCOR + VISUALIZE SCRIPT ---------------------------------------------------------------------------------------
# xcor pairing commands - [near] = 0m, 1m,..., 10m
# sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
sensor_pair_near = [(1, 7)]

dist_diff = 0
# for all sensor combination
for sensor_pair in sensor_pair_near:
    sample_no = 0
    # for all segmented signals
    for segment in n_channel_leak:
        pos1_leak_cwt, _ = pywt.cwt(segment[sensor_pair[0]], scales=scale, wavelet=m_wavelet, sampling_period=1 / fs)
        pos2_leak_cwt, _ = pywt.cwt(segment[sensor_pair[1]], scales=scale, wavelet=m_wavelet, sampling_period=1 / fs)

        # xcor for every pair of cwt
        xcor, _ = one_dim_xcor_2d_input(input_mat=np.array([pos1_leak_cwt, pos2_leak_cwt]), pair_list=[(0, 1)])
        xcor = xcor[0]
        # visualizing
        fig_title = 'Xcor of CWT of Sensor[{}] and Sensor[{}] -- Dist_Diff[{}m] -- Sample[{}]'.format(sensor_pair[0],
                                                                                                      sensor_pair[1],
                                                                                                      dist_diff,
                                                                                                      sample_no)
        fig = plot_cwt_with_time_series(time_series=[segment[sensor_pair[0]], segment[sensor_pair[1]]],
                                        no_of_time_series=2,
                                        cwt_mat=xcor,
                                        cwt_scale=scale,
                                        title=fig_title,
                                        maxpoint_searching_bound=24000)

        # only for showing the max point vector
        # mid = xcor.shape[1] // 2 + 1
        # max_xcor_vector = []
        # for row in xcor:
        #     max_along_x = np.argmax(row)
        #     max_xcor_vector.append(max_along_x - mid)
        # print(mid)
        # print(max_xcor_vector)

        plt.show()

        # saving -------------------------------------------------------------------------------------------------------
        # filename = direct_to_dir(where='result') + 'xcor_cwt_DistDiff[{}m]_sample[{}]'.format(dist_diff, sample_no)
        # fig.savefig(filename)
        # plt.close('all')
        # print('Saving --> Dist_diff: {}m, Sample: {}'.format(dist_diff, sample_no))

        sample_no += 1

    dist_diff += 1





