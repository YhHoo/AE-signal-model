import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
# self lib
from src.utils.helpers import direct_to_dir, shuffle_in_unison
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018

# data preprocessing ---------------------------------------------------------------------------------------------------
on_pc = True

if on_pc is False:
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

else:
    data = AcousticEmissionDataSet_13_7_2018(drive='F')
    dataset, label = data.leak_1bar_in_cwt_xcor_maxpoints_vector(dataset_no=2,
                                                                 f_range_to_keep=(40, 100),
                                                                 class_to_keep='all',
                                                                 shuffle=False)

# PCA OPERATION --------------------------------------------------------------------------------------------------------
no_of_pc = 5
pca = PCA(n_components=no_of_pc)  # the num of PCA
pca_result = pca.fit_transform(dataset)

print(pca_result.shape)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

pca_n_label = np.concatenate((pca_result, label.reshape((-1, 1))), axis=1)
print(pca_n_label.shape)
pca_df = pd.DataFrame(pca_n_label, columns=['pc_{}'.format(i) for i in range(no_of_pc)] + ['label'])


num_classes = 11
cmap = cm.get_cmap('rainbow')
legend_label = ['class[{}m]'.format(i) for i in range(11)]
pca_on_xaxix = 0
pca_on_yaxix = 4
# fig, ax = plt.subplots()
for i in range(num_classes):
    one_class_vec = pca_df.loc[pca_df['label'] == i].values
    plt.scatter(x=one_class_vec[:, pca_on_xaxix],
                y=one_class_vec[:, pca_on_yaxix],
                c=cmap((i+1)/num_classes),
                cmap=cm.rainbow,
                label=legend_label[i])
plt.xlabel('PCA_{}'.format(pca_on_xaxix))
plt.ylabel('PCA_{}'.format(pca_on_yaxix))
plt.legend()
plt.show()

#
# # take only first 100 data
# first_n_to_plot = 100

# # scatter plot

# for sample, sample_y in zip(dataset[:first_n_to_plot], label[:first_n_to_plot]):
#     plt.scatter(x=sample[0], y=sample[1], c=sample_y, cmap=cm.get_cmap('rainbow'))
#
# plt.show()











