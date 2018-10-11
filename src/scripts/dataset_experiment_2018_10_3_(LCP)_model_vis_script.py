import time
from src.utils.helpers import *


# file reading
dataset_filename = 'E:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/1 bar/Leak/processed/' + \
                   'lcp_recog_1bar_near_segmentation2_dataset.csv'
lcp_model = load_model(model_name='LCP_Recog_1')
lcp_model.compile(loss='binary_crossentropy', optimizer='rmsprop')


print('Reading data --> ', dataset_filename)
time_start = time.time()
data_df = pd.read_csv(dataset_filename)
print('File Read Time: {:.4f}s'.format(time.time() - time_start))
print('Full Dim: ', data_df.values.shape)

lcp_data = data_df.loc[data_df['label'] == 1].values[:, :-1]
non_lcp_data = data_df.loc[data_df['label'] == 0].values[:, :-1]

fig = plot_multiple_timeseries(input=[lcp_data[10], non_lcp_data[10]],
                               subplot_titles=['LCP', 'Non LCP'],
                               main_title='LCP and Non LCP input')

lcp_data_test = lcp_data[10].reshape((6000, 1))
non_lcp_data_test = non_lcp_data[10].reshape((6000, 1))

activation = get_activations(lcp_model, model_inputs=[lcp_data_test, non_lcp_data_test], print_shape_only=True)
print(len(activation))

# first cnn layer
activation_test = np.swapaxes(activation[5], 1, 2)

fig2 = plot_multiple_timeseries(input=activation_test[0],
                                subplot_titles=['k1', 'k2', 'k3', 'k4', 'k5'],
                                main_title='cnn1d_1 activation [LCP]')

fig3 = plot_multiple_timeseries(input=activation_test[1],
                                subplot_titles=['k1', 'k2', 'k3', 'k4', 'k5'],
                                main_title='cnn1d_1 activation [NON LCP]')

plt.show()
