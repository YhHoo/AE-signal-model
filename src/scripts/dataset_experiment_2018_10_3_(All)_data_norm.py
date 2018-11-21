'''
THIS SCRIPT NORMALIZE THE RAW LCP AND NON-LCP EXTRACTED BY data_preparation_script.py. and stored in dataset.csv,
normalize them into [-1, 1].
'''

from src.utils.helpers import *

dataset_dir = 'F:/Experiment_3_10_2018/LCP x NonLCP DATASET/'
dataset_lcp_filename = dataset_dir + 'dataset_lcp_1bar_seg4.csv'
dataset_non_lcp_filename = dataset_dir + 'dataset_non_lcp_1bar_seg1.csv'
dataset_leak_rand_filename = dataset_dir + 'dataset_leak_random_1bar.csv'
dataset_noleak_rand_filename = dataset_dir + 'dataset_noleak_random_2bar.csv'
dataset_normalized_save_filename = direct_to_dir(where='result') + 'dataset_noleak_random_2bar_norm.csv'

# change the filename to the one we wish to norm
print('Reading --> ', dataset_noleak_rand_filename)
df_data = pd.read_csv(dataset_noleak_rand_filename)
column_label = df_data.columns.values

# print(df_data)
# print(df_data.values)
# print(df_data.values.shape)

# drop rows that contains Nan
df_data.dropna(inplace=True)

df2 = pd.DataFrame(data=df_data.values, columns=column_label)
df2.to_csv(dataset_normalized_save_filename, index=False)

data_arr = df_data.values[:, :-1]
label_arr = df_data.values[:, -1].reshape(-1, 1)

temp = []
scaler = MinMaxScaler(feature_range=(-1, 1))  # ***

for row in data_arr:
    temp.append(scaler.fit_transform(row.reshape(-1, 1)).ravel())

temp = np.array(temp)

print('INPUT DATA DIM:', df_data.values.shape)

combine_arr = np.concatenate((temp, label_arr), axis=1)

print('AFTER COMBINED DIM: ', combine_arr.shape)

df_data_norm = pd.DataFrame(data=combine_arr, columns=column_label)
df_data_norm.to_csv(dataset_normalized_save_filename, index=False)