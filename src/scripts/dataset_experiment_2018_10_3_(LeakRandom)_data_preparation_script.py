import csv
import gc
from src.utils.helpers import *


# CONFIG
window_len = 6000
shuffle_tdms_seq = True
random_dataset_save_filename = direct_to_dir(where='result') + 'dataset_leak_random_2bar.csv'

# all file name
tdms_dir = 'F:/Experiment_3_10_2018/-4.5, -2, 2, 5, 8, 10, 17 (leak 1bar)/'
# tdms_dir = 'F:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/'  # discard faulty ch 20m
all_tdms_file = [(tdms_dir + f) for f in listdir(tdms_dir) if f.endswith('.tdms')]
print(len(all_tdms_file))

# shuffle
if shuffle_tdms_seq:
    all_tdms_file = np.array(all_tdms_file)[np.random.permutation(len(all_tdms_file))]

# setup header for csv
# set up a csv headers
header = np.arange(0, window_len, 1).tolist() + ['channel']

# write header to csv
with open(random_dataset_save_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

for tdms_file in all_tdms_file[:10]:
    print('Extracting --> ', tdms_file)
    n_channel_data = read_single_tdms(tdms_file)
    n_channel_data = np.swapaxes(n_channel_data, 0, 1)

    # put this line for -4.5,-2,2,5,8,17,20,23/no_leak/ data, this drop ch @ 20m
    # n_channel_data = np.delete(n_channel_data, 6, axis=0)

    print(n_channel_data.shape)

    # index for start sampling
    index = np.arange(0, n_channel_data.shape[1] - window_len, 1)
    # shuffle the item inside
    index = index[np.random.permutation(len(index))]
    for ch_no in range(6):
        temp = []
        # truncate, meaning each tdms only contribute to 20 samples
        for i in index[:1000]:
            data_in_list = n_channel_data[ch_no, i:i+window_len].tolist() + [ch_no]
            temp.append(data_in_list)

        # save to csv
        with open(random_dataset_save_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            # all ch for 1 lcp
            for entries in temp:
                writer.writerow(entries)

    print('Extraction Complete')

    gc.collect()



