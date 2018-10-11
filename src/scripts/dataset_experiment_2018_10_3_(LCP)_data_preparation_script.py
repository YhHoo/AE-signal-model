'''
THIS SCRIPT USES PEAKS INDEXES CREATED BY (LCP)_findpeak_for_all_script.py , AND SEGMENT THE AE DATA USING 6K POINTS
WINDOWS, LABELLED WITH CHANNEL NO, STORED TO LCP.CSV AND NON-LCP.CSV
'''
import gc
import csv
# self lib
from src.utils.helpers import *

# CONFIG ---------------------------------------------------------------------------------------------------------------
# operation
extract_lcp = False
extract_non_lcp = True
shuffle_tdms_list = True
max_non_lcp_sample_size = int(30e3)


# roi
roi_width = (int(1e3), int(5e3))

# save filename
lcp_dataset_save_filename = direct_to_dir(where='result') + 'dataset_lcp_2bar_near_seg3.csv'
non_lcp_dataset_save_filename = direct_to_dir(where='result') + 'dataset_non_lcp_2bar_near_seg3.csv'

# points offset when segment other channel, with respect to channel 2m and -2m
# -3m
offset_ch0 = 600
# 4m
offset_ch3 = 1100
# 6m
offset_ch4 = 2700
# 8m
offset_ch5 = 4000
# 10m
offset_ch6 = 5300
# 12m
offset_ch7 = 6500

offset_all = [offset_ch0, 0, 0, offset_ch3, offset_ch4, offset_ch5, offset_ch6, offset_ch7]

# READING LCP INDEXES --------------------------------------------------------------------------------------------------
# all file name
tdms_dir = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/2 bar/Leak/'

all_tdms_file = [(tdms_dir + f) for f in listdir(tdms_dir) if f.endswith('.tdms')]
if shuffle_tdms_list:
    random = np.random.permutation(len(all_tdms_file))
    all_tdms_file = np.array(all_tdms_file)[random]

# read the LCP indexes into df
lcp_index_filename = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/LCP DATASET/' + \
                    'lcp_index_2bar_near_segmentation3.csv'

df_1bar_lcp = pd.read_csv(lcp_index_filename, index_col=0)

# SEGMENTATION ON ALL TDMS FILES AND SAVE CSV --------------------------------------------------------------------------
# set up a csv headers
header = np.arange(0, roi_width[0]+roi_width[1], 1).tolist() + ['channel']

if extract_lcp:
    # write header to csv
    with open(lcp_dataset_save_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

if extract_non_lcp:
    # write header to csv
    with open(non_lcp_dataset_save_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

sample_size = 0
# for all tdms files
for foi in all_tdms_file:
    # get the filename e.g. test_003
    tdms_name = foi.split(sep='/')[-1]
    tdms_name = tdms_name.split(sep='.')[0]

    # locate all rows with that tdms name
    temp_df = df_1bar_lcp.loc[df_1bar_lcp['filename'] == tdms_name]

    # skip tp nex foi if df is empty
    if temp_df.empty:
        print('no LCP')
        continue

    # read all lcp indexes into a 1d array
    lcp_index_per_file = temp_df.values[:, 0]

    # read TDMS file into ram
    n_channel_data_near_leak = read_single_tdms(foi)
    n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)

    # SEGMENTING LCP ---------------------------------------------------------------------------------------------------
    if extract_lcp:
        # read all channels info into 2d array
        lcp_channel_arr = temp_df[['ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7']].values

        print('Extracting LCP --> ', lcp_index_per_file)
        # for all lcp in one file
        for lcp_index, channel in zip(lcp_index_per_file, lcp_channel_arr):

            data_list = []
            # for all valid sensor channels
            for ch_no, ch, offset in zip(np.arange(8), channel, offset_all):
                # if channel is 1 (valid)
                if ch:
                    # slicing real AE data
                    soi = n_channel_data_near_leak[ch_no, (lcp_index - roi_width[0] + offset):
                                                          (lcp_index + roi_width[1] + offset)].tolist() + [ch_no]
                    data_list.append(soi)

            # save to csv
            with open(lcp_dataset_save_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                # all ch for 1 lcp
                for row in data_list:
                    writer.writerow(row)

        print('LCP extraction completed')

    # SEGMENTING NON LCP -----------------------------------------------------------------------------------------------
    # WARNING: TERMINATE THIS CODE SOMEWHERE WHEN THE SAMPLE SIZE OF THE NON-LCP IS SUFFICIENT, LOADING ALL WILL RESULT
    # IN EXTREMELY BIG CSV, WHICH IS HARD TO LOAD INTO RAM DURING TRAINING

    if extract_non_lcp:
        # only for tdms file with >1 lcp hit
        if len(lcp_index_per_file) > 1:
            print('Extracting non-LCP --> ', lcp_index_per_file)
            # locating all indexed for non lcp
            lcp_indexes_diff = np.diff(lcp_index_per_file)

            non_lcp_indexes_per_file = []
            # for all interval btw lcp index
            for start, diff in zip(lcp_index_per_file[:-1], lcp_indexes_diff):
                allowable_segment = diff // (roi_width[1] + roi_width[0])
                if allowable_segment > 1:
                    # deciding index where nonLCP is started to be taken
                    start_index = start + roi_width[0] + roi_width[1] + offset_ch7

                    all_index = [start_index] + [(start_index + i * 6000) for i in range(1, allowable_segment - 1, 1)]
                    non_lcp_indexes_per_file.append(all_index)

            # unravel list-of-list into jz list
            non_lcp_indexes_per_file = [i for sub_list in non_lcp_indexes_per_file for i in sub_list]

            # writing segmentation to csv
            for non_lcp_indexes in non_lcp_indexes_per_file:
                data_list = []
                # for all channels
                for ch in np.arange(0, 8, 1):
                    # segment 6000 points + channel label
                    soi = n_channel_data_near_leak[ch,
                                                   non_lcp_indexes:
                                                   (non_lcp_indexes+roi_width[0]+roi_width[1])].tolist() + [ch]
                    data_list.append(soi)
                    sample_size += 1

                # save to one csv
                with open(non_lcp_dataset_save_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for row in data_list:
                        writer.writerow(row)

            print('non_lcp appended --> ', lcp_dataset_save_filename)
            print('non_lcp sample size: ', sample_size)

            # check the sample size with limit
            if sample_size > max_non_lcp_sample_size:
                print('Sample size limit achieved ! Terminating the code..')
                break

        else:
            print('Non-LCP not extracted')

    # free up memory for tdms stored
    gc.collect()


