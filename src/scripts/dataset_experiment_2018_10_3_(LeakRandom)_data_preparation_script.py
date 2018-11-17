from src.utils.helpers import *


# CONFIG
window_len = 6000
leak_random_dataset_save_filename = direct_to_dir(where='result') + 'dataset_leak_random_1bar'

# all file name
tdms_dir = 'F:/Experiment_3_10_2018/-4.5, -2, 2, 5, 8, 10, 17 (leak 1bar)/'
all_tdms_file = [(tdms_dir + f) for f in listdir(tdms_dir) if f.endswith('.tdms')]

temp = []
for f in all_tdms_file:
    n_channel_data = read_single_tdms(f)
    n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1]  # only 7 sensors
    print('TDMS data dim: ', n_channel_data.shape)

    # index for start sampling
    index = np.arange(0, n_channel_data.shape[1] - window_len, 1)
    # shuffle the item inside
    index = index[np.random.permutation(len(index))]
    # truncate, meaning each tdms only contribute to 20 samples
    for i in index[:20]:
        temp.append()