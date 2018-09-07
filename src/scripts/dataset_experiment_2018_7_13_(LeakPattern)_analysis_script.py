'''
THIS CODE IS FOR GENERATING A XCOR IMAGES USING ONLY A 5 SECONDS 8 CHANNELS DATA, FOR VISUALIZING THE DATASET BFORE
TRAINING
'''

import matplotlib.pyplot as plt
import pywt
import numpy as np
import gc
# self lib
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.helpers import direct_to_dir, read_all_tdms_from_folder, plot_cwt_with_time_series
from src.utils.dsp_tools import one_dim_xcor_2d_input, dwt_smoothing


# CONFIG --------------------------------------------------------------------------------------------------------------
# wavelet
m_wavelet = 'gaus1'
scale = np.linspace(2, 30, 100)
fs = 1e6

# segmentation
no_of_segment = 10  # 10 is showing a consistent pattern

# DATA POINT ----------------------------------------------------------------------------------------------------------
# read leak data
on_pc = True

if on_pc:
    data = AcousticEmissionDataSet_13_7_2018(drive='F')
    n_channel_leak = data.test_data(sensor_dist='near', pressure=1, leak=True)
else:
    data_dir = direct_to_dir(where='yh_laptop_test_data') + '1bar_leak/'
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
sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
# sensor_pair_near = [(1, 7)]

dist_diff = 0
# for all sensor combination
for sensor_pair in sensor_pair_near:
    sample_no = 0
    # for all segmented signals
    for segment in n_channel_leak:
        # signal denoising
        signal_1_denoised = dwt_smoothing(x=segment[sensor_pair[0]], wavelet='db4', level=3)
        signal_2_denoised = dwt_smoothing(x=segment[sensor_pair[1]], wavelet='db4', level=3)

        pos1_leak_cwt, _ = pywt.cwt(signal_1_denoised, scales=scale, wavelet=m_wavelet, sampling_period=1 / fs)
        pos2_leak_cwt, _ = pywt.cwt(signal_2_denoised, scales=scale, wavelet=m_wavelet, sampling_period=1 / fs)

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
        show_xcor = False
        if show_xcor:
            mid = xcor.shape[1] // 2 + 1
            max_xcor_vector = []
            for row in xcor:
                max_along_x = np.argmax(row)
                max_xcor_vector.append(max_along_x - mid)
            print(max_xcor_vector)

            plt.show()
            plt.close('all')

        # saving the plot ----------------------------------------------------------------------------------------------
        saving = True
        if saving:
            filename = direct_to_dir(where='google_drive') + \
                       'xcor_cwt_DistDiff[{}m]_sample[{}]'.format(dist_diff, sample_no)

            fig.savefig(filename)
            plt.close('all')
            print('Saving --> Dist_diff: {}m, Sample: {}'.format(dist_diff, sample_no))

        # update the sample no
        sample_no += 1

        # memory clean up
        # free up memory for unwanted variable
        pos1_leak_cwt, pos2_leak_cwt, xcor = None, None, None
        gc.collect()

    # update distance indicator
    dist_diff += 1





