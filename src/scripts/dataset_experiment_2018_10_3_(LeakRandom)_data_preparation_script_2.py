'''
this file is for preparing data when downsample factor is bigger than 100. Here u need to tel the downsample factor
by dsf1 x dsf2
'''

import sys
sys.path.append('C:/Users/YH/PycharmProjects/AE-signal-model')

import csv
import gc
import argparse
from scipy.signal import decimate
from src.utils.helpers import *


# -------------------------------------------------------------------------------------------------------------ARG PARSE
parser = argparse.ArgumentParser(description='Input some parameters.')
parser.add_argument('--fts', metavar='FS', default=None, type=str, help='Filename to save')
parser.add_argument('--ftr', metavar='FR', default=None, type=str, help='Filename to process')
parser.add_argument('--cth', metavar='CH', default=None, type=int, nargs='+', help='Channel no to extract')
parser.add_argument('--svs', metavar='S', default=None, type=int, help='sample vector size')
parser.add_argument('--dsf1', metavar='DF1', default=1, type=int, help='Downsample factor 1')
parser.add_argument('--dsf2', metavar='DF2', default=1, type=int, help='Downsample factor 2')

args = parser.parse_args()

# CONFIG (changes param here ONLY ***)
FILENAME_TO_SAVE = args.fts
FOLDER_TO_READ = args.ftr
CHANNEL_TO_EXTRACT = args.cth
SAMPLE_VECTOR_LENGTH = args.svs
DOWNSAMPLE_FACTOR_1 = args.dsf1
DOWNSAMPLE_FACTOR_2 = args.dsf2
SAMPLE_EXTRACTED_PER_TDMS = 150
SHUFFLE_TDMS_SEQ = True

print('Filename to save: ', FILENAME_TO_SAVE)
print('Filename to process: ', FOLDER_TO_READ)
print('Channel no to extract: ', CHANNEL_TO_EXTRACT)
print('Sample vector size: ', SAMPLE_VECTOR_LENGTH)
print('Downsample factor: {}x{}'.format(DOWNSAMPLE_FACTOR_1, DOWNSAMPLE_FACTOR_2))

# --------------------------------------------------------------------------------------------------------- FILE READING
all_tdms_file = [(FOLDER_TO_READ + f) for f in listdir(FOLDER_TO_READ) if f.endswith('.tdms')]
print('total file to extract: ', len(all_tdms_file))

# fix the random sequence
np.random.seed(43)

# shuffle
if SHUFFLE_TDMS_SEQ:
    all_tdms_file = np.array(all_tdms_file)[np.random.permutation(len(all_tdms_file))]

# setup header for csv
# set up a csv headers
header = np.arange(0, SAMPLE_VECTOR_LENGTH, 1).tolist() + ['channel']

# write header to csv
with open(FILENAME_TO_SAVE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

file_count = 1
for tdms_file in all_tdms_file:
    print('File No: ', file_count)
    n_channel_data = read_single_tdms(tdms_file)
    n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1, :]  # drop last channel, due to no sensor

    temp, temp2 = [], []

    # first downsample
    for channel in n_channel_data:
        temp.append(decimate(x=channel, q=DOWNSAMPLE_FACTOR_1))
    # second downsample
    for channel in temp:
        temp2.append(decimate(x=channel, q=DOWNSAMPLE_FACTOR_2))

    n_channel_data = np.array(temp2)
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
        print('Extracting channel {} --> {} samples'.format(ch_no, SAMPLE_EXTRACTED_PER_TDMS))
        temp = []
        # truncate, meaning each tdms only contribute to 20 samples
        for i in index[:SAMPLE_EXTRACTED_PER_TDMS]:
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

    file_count += 1



