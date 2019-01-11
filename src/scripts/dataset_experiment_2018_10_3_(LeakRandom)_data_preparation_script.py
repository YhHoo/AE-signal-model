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
shuffle_tdms_seq = True
downsample_factor_status = True
if DOWNSAMPLE_FACTOR is 1:
    downsample_factor_status = False


# all file name
all_tdms_file = [(FOLDER_TO_READ + f) for f in listdir(FOLDER_TO_READ) if f.endswith('.tdms')]
print('total file to extract: ', len(all_tdms_file))

# shuffle
if shuffle_tdms_seq:
    all_tdms_file = np.array(all_tdms_file)[np.random.permutation(len(all_tdms_file))]

# setup header for csv
# set up a csv headers
header = np.arange(0, SAMPLE_VECTOR_LENGTH, 1).tolist() + ['channel']

# write header to csv
with open(FILENAME_TO_SAVE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

for tdms_file in all_tdms_file:
    n_channel_data = read_single_tdms(tdms_file)
    n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1, :]  # drop last channel, due to no sensor

    temp = []
    if downsample_factor_status:
        for channel in n_channel_data:
            temp.append(decimate(x=channel, q=DOWNSAMPLE_FACTOR))
        n_channel_data = np.array(temp)
        print('Dim After Downsample: ', n_channel_data.shape)

    # put this line for -4.5,-2,2,5,8,17,20,23/no_leak/ data, this drop ch @ 20m
    # n_channel_data = np.delete(n_channel_data, 3, axis=0)

    print('Dim before extraction: ', n_channel_data.shape)

    # index for start sampling
    index = np.arange(0, n_channel_data.shape[1] - SAMPLE_VECTOR_LENGTH, 1)

    # shuffle the item inside
    index = index[np.random.permutation(len(index))]

    # for all channels or for specific channel only
    for ch_no in CHANNEL_TO_EXTRACT:
        print('Extracting channel {}'.format(ch_no))
        temp = []
        # truncate, meaning each tdms only contribute to 20 samples
        for i in index[:150]:
            data_in_list = n_channel_data[ch_no, i:i + SAMPLE_VECTOR_LENGTH].tolist() + [ch_no]
            temp.append(data_in_list)

        # save to csv
        with open(FILENAME_TO_SAVE, 'a', newline='') as f:
            writer = csv.writer(f)
            # all ch for 1 lcp
            for entries in temp:
                writer.writerow(entries)

    print('Extraction Complete')

    gc.collect()



