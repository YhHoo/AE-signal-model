import pywt
import time
import pandas as pd
import numpy as np
import multiprocessing as mp
from src.utils.dsp_tools import one_dim_xcor_2d_input
from src.utils.helpers import direct_to_dir, read_single_tdms, ProgressBarForLoop


def multiprocess(n_channel_segment):
    '''
    :param n_channel_segment: a 2d array, where rows is scale, column is xcor step
    :param sensor_pair: a tuple
    :return: a 1d feature vector
    '''
    # wavelet
    m_wavelet = 'gaus1'
    scale = np.linspace(2, 30, 50)
    sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]

    all_max_xcor_vector = []
    for sensor_pair in sensor_pair_near:
        # cwt
        pos1_leak_cwt, _ = pywt.cwt(n_channel_segment[sensor_pair[0]], scales=scale, wavelet=m_wavelet)
        pos2_leak_cwt, _ = pywt.cwt(n_channel_segment[sensor_pair[1]], scales=scale, wavelet=m_wavelet)

        xcor, _ = one_dim_xcor_2d_input(input_mat=np.array([pos1_leak_cwt, pos2_leak_cwt]),
                                        pair_list=[(0, 1)])
        xcor = xcor[0]
        # midpoint in xcor
        mid = xcor.shape[1] // 2 + 1

        max_xcor_vector = []
        # for every row of xcor, find max point index
        for row in xcor:
            max_along_x = np.argmax(row)
            max_xcor_vector.append(max_along_x - mid)

        all_max_xcor_vector.append(max_xcor_vector)

    return all_max_xcor_vector


if __name__ == '__main__':
    # wavelet
    m_wavelet = 'gaus1'
    scale = np.linspace(2, 30, 50)
    fs = 1e6

    # creating dict to store each class data
    all_class = {}
    for i in range(0, 11, 1):
        all_class['class_[{}]'.format(i)] = []

    tdms_dir = direct_to_dir(where='yh_laptop_test_data') + '/1bar_leak/test_0001.tdms'
    # read raw from drive
    n_channel_data_near_leak = read_single_tdms(tdms_dir)
    n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)

    # split on time axis into no_of_segment
    n_channel_data_near_leak = np.split(n_channel_data_near_leak, axis=1, indices_or_sections=50)

    print('RUNNING')
    start_time = time.time()
    pool = mp.Pool()
    max_vector_list = pool.map(multiprocess, n_channel_data_near_leak)
    max_vector_list = np.array(max_vector_list)
    # max_vector_list gives [no_of_segment, no_of_sensor_pair, vector_points]
    max_vector_list = np.array(max_vector_list)
    print(max_vector_list.shape)
    max_vector_list = np.swapaxes(max_vector_list, 0, 1)
    print(max_vector_list.shape)

    # append the result to all_class dict (expect to get dim of (11, 50, 50)-->(sensor pair, samples, feature point))
    for i in range(0, 11, 1):
        all_class['class_[{}]'.format(i)].append(max_vector_list[i])

    # just to display the dict full dim
    temp = []
    for _, value in all_class.items():
        temp.append(value[0])
    temp = np.array(temp)
    print('all_class dim: ', temp.shape)

    print('Exec Time: {:.3f} s'.format(time.time() - start_time))  # this pool code takes 18mins

    dataset = []
    label = []
    for i in range(0, 11, 1):
        # for all samples in each class
        for sample in all_class['class_[{}]'.format(i)][0]:
            dataset.append(sample)
            label.append(i)

    # convert to array
    dataset = np.array(dataset)
    label = np.array(label)
    print('Dataset Dim: ', dataset.shape)
    print('Label Dim: ', label.shape)

    # save to csv
    label = label.reshape((-1, 1))
    all_in_one = np.concatenate([dataset, label], axis=1)
    # column label
    freq = pywt.scale2frequency(wavelet=m_wavelet, scale=scale) * fs
    column_label = ['Scale_{:.4f}_Freq_{:.4f}Hz'.format(i, j) for i, j in zip(scale, freq)] + ['label']
    df = pd.DataFrame(all_in_one, columns=column_label)
    filename = direct_to_dir(where='result') + 'cwt_xcor_maxpoints_vector_dataset_{}.csv'.format(2)
    df.to_csv(filename)





