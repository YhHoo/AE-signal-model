import time
from src.utils.helpers import *
from src.utils.dsp_tools import *


# file reading
dataset_dir = 'C:/Users/YH/Desktop/LCP DATASET/'
lcp_dataset_filename = dataset_dir + 'dataset_lcp_1bar_seg4_norm.csv'
non_lcp_dataset_filename = dataset_dir + 'dataset_non_lcp_1bar_seg1_norm.csv'

lcp_model = load_model(model_name='LCP_Recog_1')
lcp_model.compile(loss='binary_crossentropy', optimizer='rmsprop')


print('Reading data --> ', lcp_dataset_filename)
time_start = time.time()
lcp_df = pd.read_csv(lcp_dataset_filename)
print('File Read Time: {:.4f}s'.format(time.time() - time_start))
print('Full Dim: ', lcp_df.values.shape)

non_lcp_df = pd.read_csv(non_lcp_dataset_filename)
print('File Read Time: {:.4f}s'.format(time.time() - time_start))
print('Full Dim: ', non_lcp_df.values.shape)

lcp_data = lcp_df.loc[lcp_df['channel'] == 1].values[:, :-1]
non_lcp_data = non_lcp_df.loc[non_lcp_df['channel'] == 1].values[:, :-1]

# shuffle lcp and nonlcp data entries
lcp_data = lcp_data[np.random.permutation(len(lcp_data))]
non_lcp_data = non_lcp_data[np.random.permutation(len(non_lcp_data))]


# take first 10 samples of LCP and nonLCP
for sample_no in range(0, 10, 1):
    layer_no_to_visualize = 0
    activation = get_activations(lcp_model,
                                 model_inputs=[lcp_data[sample_no].reshape((6000, 1)),
                                               non_lcp_data[sample_no].reshape((6000, 1))],
                                 print_shape_only=True)
    activation = np.swapaxes(activation[layer_no_to_visualize], 1, 2)

    for filter_no in range(0, activation.shape[1], 1):
        # FFT of input
        lcp_fft_y, _, lcp_fft_x = fft_scipy(sampled_data=lcp_data[sample_no], visualize=False, fs=1e6)
        nonlcp_fft_y, _, nonlcp_fft_x = fft_scipy(sampled_data=non_lcp_data[sample_no], visualize=False, fs=1e6)

        # FFT of activation
        lcp_act_fft_y, _, lcp_act_fft_x = fft_scipy(sampled_data=activation[0, filter_no, :], visualize=False, fs=1e6)
        nonlcp_act_fft_y, _, nonlcp_act_fft_x = fft_scipy(sampled_data=activation[1, filter_no, :],
                                                          visualize=False, fs=1e6)

        fig_all = plt.figure(figsize=(13, 8))
        fig_all.suptitle('Model Vis, Sample[{}]_Layer[{}]_Filter[{}]'.format(sample_no,
                                                                             layer_no_to_visualize,
                                                                             filter_no),
                         fontweight='bold')
        fig_all.subplots_adjust(hspace=0.51, bottom=0.06, top=0.92)
        ax_input_lcp = fig_all.add_subplot(4, 2, 1)
        ax_input_lcp.set_title('Input LCP')
        ax_act_lcp = fig_all.add_subplot(4, 2, 3)
        ax_act_lcp.set_title('Activation LCP')
        ax_fft_lcp = fig_all.add_subplot(4, 2, 5)
        ax_fft_lcp.set_title('FFT LCP Input')
        ax_fft_lcp_act = fig_all.add_subplot(4, 2, 7, sharex=ax_fft_lcp)
        ax_fft_lcp_act.set_title('FFT Activation LCP')

        ax_input_nonlcp = fig_all.add_subplot(4, 2, 2)
        ax_input_nonlcp.set_title('Input NonLCP')
        ax_act_nonlcp = fig_all.add_subplot(4, 2, 4)
        ax_act_nonlcp.set_title('Activation NonLCP')
        ax_fft_nonlcp = fig_all.add_subplot(4, 2, 6)
        ax_fft_nonlcp.set_title('FFT NonLCP Input')
        ax_fft_nonlcp_act = fig_all.add_subplot(4, 2, 8, sharex=ax_fft_nonlcp)
        ax_fft_nonlcp_act.set_title('FFT Activation NonLCP')

        # LCP
        ax_input_lcp.plot(lcp_data[sample_no])
        ax_act_lcp.plot(activation[0, filter_no, :])
        ax_fft_lcp.plot(lcp_fft_x[2:], lcp_fft_y[2:])  # to avoid those spiking boundary effect
        ax_fft_lcp.set_xlabel('Hz')
        ax_fft_lcp_act.plot(lcp_act_fft_x[2:], lcp_act_fft_y[2:])
        ax_fft_lcp_act.set_xlabel('Hz')

        # nonLCP
        ax_input_nonlcp.plot(non_lcp_data[sample_no])
        ax_act_nonlcp.plot(activation[1, filter_no, :])
        ax_fft_nonlcp.plot(nonlcp_fft_x[2:], nonlcp_fft_y[2:])
        ax_fft_nonlcp.set_xlabel('Hz')
        ax_fft_nonlcp_act.plot(nonlcp_act_fft_x[2:], nonlcp_act_fft_y[2:])
        ax_fft_nonlcp_act.set_xlabel('Hz')

        plt.show()


# fig = plot_multiple_timeseries(input=[lcp_data[10], non_lcp_data[10]],
#                                subplot_titles=['LCP', 'Non LCP'],
#                                main_title='LCP and Non LCP input')

# lcp_data_test = lcp_data[10].reshape((6000, 1))
# non_lcp_data_test = non_lcp_data[10].reshape((6000, 1))
#
# activation = get_activations(lcp_model, model_inputs=[lcp_data_test, non_lcp_data_test], print_shape_only=True)
#

# # first cnn layer
# activation_test = np.swapaxes(activation[0], 1, 2)
#
# fig2 = plot_multiple_timeseries(input=activation_test[0, :5],
#                                 subplot_titles=['k1', 'k2', 'k3', 'k4', 'k5'],
#                                 main_title='cnn1d_1 activation [LCP]')
#
# fig3 = plot_multiple_timeseries(input=activation_test[1, :5],
#                                 subplot_titles=['k1', 'k2', 'k3', 'k4', 'k5'],
#                                 main_title='cnn1d_1 activation [NON LCP]')
#
# plt.show()


