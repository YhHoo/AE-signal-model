import numpy as np
import pandas as pd
from os import listdir
import gc
import csv
# self lib
from src.utils.helpers import read_single_tdms, direct_to_dir, plot_multiple_timeseries, lollipop_plot

# CONFIG ---------------------------------------------------------------------------------------------------------------
# roi
roi_width = (int(1e3), int(5e3))
lcp_recognition_dataset_save_filename = direct_to_dir(where='result') + 'lcp_recog_1bar_near_segmentation2_dataset.csv'
# non_lcp_save_filename = direct_to_dir(where='result') + 'non_lcp_1bar_near_segmentation2_dataset.csv'

# READING LCP INDEXES --------------------------------------------------------------------------------------------------
# all file name
tdms_dir = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/Leak/'
all_tdms_dir = [(tdms_dir + f) for f in listdir(tdms_dir) if f.endswith('.tdms')]

# read the LCP into df
lcp_dir = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/Leak/processed/' + \
          'lcp_index_1bar_near_segmentation2.csv'
lcp_df = pd.read_csv(lcp_dir, index_col=0)

# get only confident LCP rows
lcp_confident_df = lcp_df.loc[lcp_df['confident LCP'] == 1]
lcp_confident_df = lcp_confident_df.drop(['contain other source'], axis=1)


# SEGMENTATION ON ALL TDMS FILES AND SAVE CSV --------------------------------------------------------------------------
# set up a csv headers
header = np.arange(0, roi_width[0]+roi_width[1], 1).tolist() + ['label']

# write first row to csv
with open(lcp_recognition_dataset_save_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

# lcp_signal, non_lcp_signal = [], []
for foi in all_tdms_dir:
    # get the filename e.g. test_003.tdms
    tdms_name = foi.split(sep='/')[-1]
    tdms_name = tdms_name.split(sep='.')[0]

    # locate all rows with that tdms name
    temp_df = lcp_confident_df.loc[lcp_confident_df['filename'] == tdms_name]
    lcp_index_per_file = temp_df.values[:, 0]

    # for empty array, skip to nex foi
    if len(lcp_index_per_file) is 0:
        print('no LCP')
        continue

    print(lcp_index_per_file)

    n_channel_data_near_leak = read_single_tdms(foi)
    n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)

    # SEGMENTING LCP ---------------------------------------------------------------------------------------------------
    for lcp_index in lcp_index_per_file:
        # segment sensor -2m
        soi_1 = n_channel_data_near_leak[1, (lcp_index - roi_width[0]):(lcp_index + roi_width[1])].tolist() + [1]
        # segment sensor 2m
        soi_2 = n_channel_data_near_leak[2, (lcp_index - roi_width[0]):(lcp_index + roi_width[1])].tolist() + [1]

        # save to csv
        with open(lcp_recognition_dataset_save_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(soi_1)
            writer.writerow(soi_2)

    print('lcp appended --> ', lcp_recognition_dataset_save_filename)

    # SEGMENTING NON LCP -----------------------------------------------------------------------------------------------
    # only for tdms file with >1 lcp hit
    if len(lcp_index_per_file) > 1:
        # locating all indexed for non lcp
        lcp_indexes_diff = np.diff(lcp_index_per_file)
        non_lcp_indexes_per_file = []

        # for all interval btw lcp index
        for start, diff in zip(lcp_index_per_file[:-1], lcp_indexes_diff):
            allowable_segment = diff // (roi_width[1] + roi_width[0])
            if allowable_segment > 1:
                start_index = start + roi_width[0] + roi_width[1]

                all_index = [start_index] + [(start_index + i * 6000) for i in range(1, allowable_segment - 1, 1)]
                non_lcp_indexes_per_file.append(all_index)

        non_lcp_indexes_per_file = [i for sub_list in non_lcp_indexes_per_file for i in sub_list]

        # writing segmentation to csv
        for non_lcp_indexes in non_lcp_indexes_per_file:
            # segment 6000 points
            soi = n_channel_data_near_leak[1, (non_lcp_indexes-roi_width[0]):(non_lcp_indexes+roi_width[1])].tolist() \
                  + [0]

            # save to one csv
            with open(lcp_recognition_dataset_save_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(soi)

        print('non_lcp appended --> ', lcp_recognition_dataset_save_filename)

    gc.collect()


