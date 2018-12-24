'''
THIS SCRIPT IS FOR VISUALIZING THROUGH ALL LCP AND NONLCP EXTRACTED BY find_peak_for_all_script.py and
data_preparation_script.py.
'''

from src.utils.helpers import *


dataset_dir = 'F:/Experiment_3_10_2018/LCP x NonLCP DATASET/'
dataset_to_visualize = dataset_dir + 'dataset_leak_random_1bar_3.csv'

input_data_labels = ['sensor@[-4.5m]',  # the channels' dist of the input data
                     'sensor@[-2m]',
                     'sensor@[2m]',
                     'sensor@[5m]',
                     'sensor@[8m]',
                     'sensor@[10m]']

print('Reading --> ', dataset_to_visualize)
data_df = pd.read_csv(dataset_to_visualize)
print(data_df.head())
print(data_df.values.shape)

scaler = MinMaxScaler(feature_range=(-1, 1))

for ch in range(0, 6, 1):
    selected_ch_df = data_df[data_df['channel'] == ch]
    print('ch_{} sample size: {}'.format(ch, len(selected_ch_df.values)))
    temp_arr = selected_ch_df.values[:, :-1]
    # random selection
    rand_index = np.random.permutation(len(temp_arr))
    rand_sample_to_plot = temp_arr[rand_index[:16]]

    # finding FFT of 5 samples
    fig_fft = plt.figure(figsize=(14, 8))
    fig_fft.suptitle('FFT_{}_[{}]'.format(dataset_to_visualize, input_data_labels[ch]), fontweight='bold')
    ax_fft = fig_fft.add_subplot(1, 1, 1)
    ax_fft.grid('on')
    for sample in rand_sample_to_plot[:10]:
        f_mag_unseen, _, f_axis = fft_scipy(sampled_data=sample, fs=int(1e6), visualize=False)
        ax_fft.plot(f_axis[10:], f_mag_unseen[10:], alpha=0.5)

    # real plot
    fig_raw = plt.figure(figsize=(14, 8))
    fig_raw.subplots_adjust(left=0.09, right=0.93, bottom=0.07, top=0.91, hspace=0.37)
    fig_raw.suptitle('RAW_{}_[{}]'.format(dataset_to_visualize, input_data_labels[ch]), fontweight='bold')
    ylim = 1
    for i, lcp in zip(np.arange(1, 17, 1), rand_sample_to_plot):
        ax_lcp = fig_raw.add_subplot(4, 4, i)

        # normalize every lcp
        lcp_normalized = scaler.fit_transform(lcp.reshape(-1, 1))
        ax_lcp.plot(lcp_normalized)
        ax_lcp.set_title('Sample_{}'.format(i))
        ax_lcp.set_ylim(bottom=-ylim, top=ylim)

    fig_save_filename = direct_to_dir(where='result') + 'Data_fft_CH{}.png'.format(ch)
    fig_save_filename_2 = direct_to_dir(where='result') + 'Data_vis_CH{}.png'.format(ch)
    fig_fft.savefig(fig_save_filename)
    fig_raw.savefig(fig_save_filename_2)

    plt.close('all')
    # plt.show()

# print('Reading --> ', dataset_non_lcp_filename)
# non_lcp_df = pd.read_csv(dataset_non_lcp_filename)
# print(non_lcp_df.head())
