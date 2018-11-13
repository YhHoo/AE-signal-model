import time
from src.utils.helpers import *
from src.utils.dsp_tools import *


# file reading
dataset_dir_in_laptop = 'C:/Users/YH/Desktop/LCP DATASET/'
dataset_dir_in_pc = 'F:/Experiment_3_10_2018/LCP x NonLCP DATASET/'
lcp_dataset_filename = dataset_dir_in_pc + 'dataset_lcp_1bar_seg4_norm.csv'
non_lcp_dataset_filename = dataset_dir_in_pc + 'dataset_non_lcp_1bar_seg1_norm.csv'

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
    layer_no_to_visualize = 8
    activation = get_activations(lcp_model,
                                 model_inputs=[lcp_data[sample_no].reshape((6000, 1)),
                                               non_lcp_data[sample_no].reshape((6000, 1))],
                                 print_shape_only=True)
    # activation = activation[layer_no_to_visualize]
    activation = np.swapaxes(activation[layer_no_to_visualize], 1, 2)
    # print('filter[{} ] = LCP |  NonLCP')
    for filter_no in range(0, activation.shape[1], 1):
        # print('filter[{} ] = {:.4f}  |  {:.4f}'.
        # format(filter_no, activation[0, filter_no],  activation[1, filter_no]))

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

        fig_save_filename = direct_to_dir(where='result') + 'LCP_Recog_1_lyr[{}]_fl[{}]_sample[{}]'.\
            format(layer_no_to_visualize, filter_no, sample_no)
        fig_all.savefig(fig_save_filename)
        print('saving --> ', fig_save_filename)
        plt.close('all')



