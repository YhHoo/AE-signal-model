import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# self lib
from src.utils.helpers import direct_to_dir, shuffle_in_unison

# data preprocessing ---------------------------------------------------------------------------------------------------
f_range_to_keep = (40, 100)

filename = direct_to_dir(where='result') + 'test.csv'
data_df = pd.read_csv(filename)
data_df_col_name = data_df.columns[1:-1]

# convert df values to arrays
data_mat = data_df.values

# drop the first column, segment the 2d mat into dataset and label
dataset = data_mat[:, 1:-1]
label = data_mat[:, -1]

dataset = dataset[:, f_range_to_keep[0]:f_range_to_keep[1]]

# std normalize the data
dataset_shape = dataset.shape
scaler = StandardScaler()
dataset = scaler.fit_transform(dataset.ravel().reshape(-1, 1).astype('float64'))
dataset = dataset.reshape(dataset_shape)

# shuffle data set
dataset, label = shuffle_in_unison(dataset, label)

print(dataset.shape)
print(label.shape)

# PCA OPERATION --------------------------------------------------------------------------------------------------------
pca = PCA(n_components=3)  # the num of PCA
pca_result = pca.fit_transform(dataset)

print(pca_result.shape)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# take only first 100 data
first_n_to_plot = 100

# scatter plot
legend_label = ['class[{}m]'.format(i) for i in range(11)]
for sample, sample_y in zip(dataset[:first_n_to_plot], label[:first_n_to_plot]):
    plt.scatter(x=sample[0], y=sample[1], c=sample_y, cmap=cm.get_cmap('rainbow'))

plt.show()