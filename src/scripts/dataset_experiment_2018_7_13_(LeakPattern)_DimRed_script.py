'''
This script applies dimensionality reduction on the multidimensional feature vectors, using T-sne or PCA, followed by
scatter plot visualization
'''
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# self lib
from src.utils.helpers import direct_to_dir, shuffle_in_unison, scatter_plot, scatter_plot_3d_vispy
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018

# data preprocessing ---------------------------------------------------------------------------------------------------
on_pc = True

if on_pc is False:
    f_range_to_keep = (0, 50)

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
    dataset, label = data.leak_1bar_in_cwt_xcor_maxpoints_vector(dataset_name='bounded_xcor_3',
                                                                 f_range_to_keep=(0, 100),
                                                                 class_to_keep='all',
                                                                 shuffle=True)

# PCA OPERATION --------------------------------------------------------------------------------------------------------
pca_op = False

if pca_op:
    pca = PCA(n_components=3)  # the num of PCA
    reduced_result = pca.fit_transform(dataset)

    print('PCA output DIM: ', reduced_result.shape)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# T-sne Operation ------------------------------------------------------------------------------------------------------
tsne_op = True

per = [70, 100, 120, 150]
count = 0
if tsne_op:
    for p in per:
        time_start = time.time()
        # the more complex the data, set perplexity higher
        tsne = TSNE(n_components=2, verbose=1, perplexity=p, n_iter=3000)
        reduced_result = tsne.fit_transform(dataset)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
        print('TSNE output DIM: ', reduced_result.shape)

        fig = scatter_plot(dataset=reduced_result, label=label, num_classes=11, feature_to_plot=(0, 1),
                           annotate_all_point=True,
                           title='cwt_xcor_maxpoints_vector_dataset_bounded_xcor_3_(TSNE_5k_epoch_{}_per)'.format(p),
                           save_data_to_csv=True)

        # saving
        fig_filename = direct_to_dir(where='result') + 'tsne_{}'.format(count)
        fig.savefig(fig_filename)

        count += 1

        plt.close('all')


# visualize in scatter plot --------------------------------------------------------------------------------------------

# for points less than 1k
# fig = scatter_plot(dataset=reduced_result, label=label, num_classes=11, feature_to_plot=(0, 1),
#                    annotate_all_point=True,
#                    title='cwt_xcor_maxpoints_vector_dataset_bounded_xcor_3_(TSNE_5k_epoch_70_per)',
#                    save_data_to_csv=True)
#
# plt.show()
# saving
# fig_filename = direct_to_dir(where='result') + 'scatter_plot_(bounded_xcor)'
# fig.savefig(fig_filename)

# for 3d points more than 1k
# scatter_plot_3d_vispy(dataset=reduced_result, label=label)












