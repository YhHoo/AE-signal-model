'''
THIS SCRIPT NORMALIZE THE RAW LCP AND NON-LCP EXTRACTED BY data_preparation_script.py. and stored in dataset.csv,
normalize them into [-1, 1].
'''

from src.utils.helpers import *

dataset_dir = 'F:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/leak_noleak_preprocessed_dataset/'  # *
filename = 'dataset_noleak_random_2bar_[4]'  # *

dataset_to_norm = dataset_dir + filename + '.csv'
dataset_norma_save_filename = dataset_dir + '{}_norm.csv'.format(filename)

# change the filename to the one we wish to norm
print('Reading --> ', dataset_to_norm)
df_data = pd.read_csv(dataset_to_norm)
column_label = df_data.columns.values
print('INPUT DATA DIM:', df_data.values.shape)

# drop rows that contains Nan
df_data.dropna(inplace=True)

data_arr = df_data.values[:, :-1]
label_arr = df_data.values[:, -1].reshape(-1, 1)

temp = []
scaler = MinMaxScaler(feature_range=(-1, 1))  # ***

for row in data_arr:
    temp.append(scaler.fit_transform(row.reshape(-1, 1)).ravel())

temp = np.array(temp)

combine_arr = np.concatenate((temp, label_arr), axis=1)

print('AFTER COMBINED DIM: ', combine_arr.shape)

df_data_norm = pd.DataFrame(data=combine_arr, columns=column_label)
df_data_norm.to_csv(dataset_norma_save_filename, index=False)
print('Saved -> ', dataset_norma_save_filename)
