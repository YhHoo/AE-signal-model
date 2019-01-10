import sys
sys.path.append('C:/Users/YH/PycharmProjects/AE-signal-model')

import csv
import gc
import argparse
from scipy.signal import decimate
from src.utils.helpers import *


# argparse
parser = argparse.ArgumentParser(description='Input some parameters.')
parser.add_argument('--fts', metavar='FS', default=None, type=str, help='Filename to save')
parser.add_argument('--ftr', metavar='FR', default=None, type=str, help='Filename to process')
parser.add_argument('--cth', metavar='CH', default=None, type=int, nargs='+', help='Channel no to extract')
parser.add_argument('--svs', metavar='S', default=None, type=int, help='sample vector size')
parser.add_argument('--dsf', metavar='DF', default=1, type=int, help='Downsample factor')

args = parser.parse_args()

FILENAME_TO_SAVE = args.fts
FOLDER_TO_READ = args.ftr
CHANNEL_TO_EXTRACT = args.cth
SAMPLE_VECTOR_LENGTH = args.svs
DOWNSAMPLE_FACTOR = args.dsf

print('Filename to save: ', FILENAME_TO_SAVE)
print('Filename to process: ', FOLDER_TO_READ)
print('Channel no to extract: ', CHANNEL_TO_EXTRACT)
print('sample vector size: ', SAMPLE_VECTOR_LENGTH)
print('Downsample factor: ', DOWNSAMPLE_FACTOR)


# CONFIG
channels_to_take = CHANNEL_TO_EXTRACT
sample_vector_size = SAMPLE_VECTOR_LENGTH
shuffle_tdms_seq = True
downsample_factor = DOWNSAMPLE_FACTOR
downsample_factor_status = True
if downsample_factor is 1:
    downsample_factor_status = False


random_dataset_save_filename = direct_to_dir(where='result') + 'dataset_leak_random_1.5bar_[0]_ds.csv'

# all file name
tdms_dir = 'G:/Experiment_3_1_2019/-3,-2,0,5,7,16,17/1.5 bar/Leak/Train & Val data/'
# tdms_dir = 'F:/Experiment_21_12_2018/8Ch/-3,-2,0,5,7,15,16/2 bar/Leak/Train & Val data/'
# tdms_dir = 'F:/Experiment_3_10_2018/-4.5, -2, 2, 5, 8, 10, 17 (leak 1bar)/'
# tdms_dir = 'F:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/'  # discard faulty ch 20m
all_tdms_file = [(tdms_dir + f) for f in listdir(tdms_dir) if f.endswith('.tdms')]
print('total file to extract: ', len(all_tdms_file))

# shuffle
if shuffle_tdms_seq:
    all_tdms_file = np.array(all_tdms_file)[np.random.permutation(len(all_tdms_file))]

# setup header for csv
# set up a csv headers
header = np.arange(0, sample_vector_size, 1).tolist() + ['channel']

# write header to csv
with open(random_dataset_save_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

for tdms_file in all_tdms_file:
    n_channel_data = read_single_tdms(tdms_file)
    n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1, :]  # drop last channel, due to no sensor

    temp = []
    if downsample_factor_status:
        for channel in n_channel_data:
            temp.append(decimate(x=channel, q=downsample_factor))
        n_channel_data = np.array(temp)
        print('Dim After Downsample: ', n_channel_data.shape)

    # put this line for -4.5,-2,2,5,8,17,20,23/no_leak/ data, this drop ch @ 20m
    # n_channel_data = np.delete(n_channel_data, 3, axis=0)

    print('Dim before extraction: ', n_channel_data.shape)

    # index for start sampling
    index = np.arange(0, n_channel_data.shape[1] - sample_vector_size, 1)
    # shuffle the item inside
    index = index[np.random.permutation(len(index))]

    # for all channels or for specific channel only
    for ch_no in channels_to_take:
        print('Extracting channel {}'.format(ch_no))
        temp = []
        # truncate, meaning each tdms only contribute to 20 samples
        for i in index[:150]:
            data_in_list = n_channel_data[ch_no, i:i + sample_vector_size].tolist() + [ch_no]
            temp.append(data_in_list)

        # save to csv
        with open(random_dataset_save_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            # all ch for 1 lcp
            for entries in temp:
                writer.writerow(entries)

    print('Extraction Complete')

    gc.collect()



