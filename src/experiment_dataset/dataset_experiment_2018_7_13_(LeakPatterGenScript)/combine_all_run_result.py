'''
This script is to combine all 3 csv of max vector generated in to one. All 3 csv must contains all same classes.
This script will gather samples of same class and append into a continuous rows of vector.
'''
import numpy as np
import pandas as pd
from src.utils.helpers import direct_to_dir

# change the filename here only
folder_dir = direct_to_dir(where='result') + 'cwt_xcor_maxpoints_vector_dataset_sliced_xcor'
filename_1 = folder_dir + '_p1.csv'
filename_2 = folder_dir + '_p2.csv'
filename_3 = folder_dir + '_p3.csv'

all_filename = [filename_1, filename_2, filename_3]

file_1_df = pd.read_csv(filename_1)
file_2_df = pd.read_csv(filename_1)
file_3_df = pd.read_csv(filename_1)

# find unique labels in data 'label' column
all_label_1 = np.unique(file_1_df.values[:, -1])
all_label_2 = np.unique(file_2_df.values[:, -1])
all_label_3 = np.unique(file_3_df.values[:, -1])

assert all_label_1 == all_label_2 == all_label_3, 'Labels contained in all 3 files are different'

list_of_array = []
# for all classes
for label in all_label_1:
    # extract same classes from all 3 csv
    mat1 = file_1_df.loc[file_1_df['label'] == label].values[:, 1:]  # discard first column of index
    mat2 = file_2_df.loc[file_2_df['label'] == label].values[:, 1:]
    mat3 = file_3_df.loc[file_3_df['label'] == label].values[:, 1:]
    list_of_array.append(np.concatenate([mat1, mat2, mat3], axis=0))

dataset_combined = np.concatenate(list_of_array, axis=0)
print(dataset_combined.shape)

# # create a dict
# all_class = {}
# for i in range(0, 11, 1):
#     all_class['class_[{}]'.format(i)] = []
#
# # Extracting all data from 3 csv
# # for all 3 csv
# for filename in all_filename:
#     # for all labels
#     for label in all_label:
#         # group all rows of specific labels
#         same_label_mat = file_df.loc[file_df['label'] == label].values[:, 1:]
#         all_class['class_[{}]'.format(label)].append(same_label_mat)
#
# # Storing the data extracted into new storage
# # for all class
# for label in all_label:


