'''
THIS SCRIPT IS FOR VISUALIZING THROUGH ALL LCP AND NONLCP EXTRACTED BY find_peak_for_all_script.py and
data_preparation_script.py.
'''

from src.utils.helpers import *


dataset_dir = 'F:/Experiment_3_10_2018/LCP x NonLCP DATASET/'
dataset_lcp_filename = dataset_dir + 'dataset_lcp_1bar_seg4.csv'
dataset_non_lcp_filename = dataset_dir + 'dataset_non_lcp_1bar_seg_1.csv'

print('Reading --> ', dataset_non_lcp_filename)
lcp_df = pd.read_csv(dataset_non_lcp_filename)
print(lcp_df.head())
print(lcp_df.values.shape)

scaler = MinMaxScaler(feature_range=(-1, 1))

for ch in range(0, 6, 1):
    lcp_selected_ch_df = lcp_df[lcp_df['channel'] == ch]
    temp_arr = lcp_selected_ch_df.values[:, :-1]
    # random selection
    rand_index = np.random.permutation(len(temp_arr))
    rand_lcp_to_plot = temp_arr[rand_index[:16]]

    fig_lcp = plt.figure(figsize=(14, 8))
    fig_lcp.subplots_adjust(left=0.09, right=0.93, bottom=0.07, top=0.91, hspace=0.37)
    fig_lcp.suptitle('Non LCP CH{}_normalized RANDOM'.format(ch), fontweight='bold')
    ylim = 1
    for i, lcp in zip(np.arange(1, 17, 1), rand_lcp_to_plot):
        ax_lcp = fig_lcp.add_subplot(4, 4, i)

        # normalize every lcp
        lcp_normalized = scaler.fit_transform(lcp.reshape(-1, 1))
        ax_lcp.plot(lcp_normalized)
        ax_lcp.set_title('Non_LCP_{}'.format(i))
        ax_lcp.set_ylim(bottom=-ylim, top=ylim)

    fig_save_filename = direct_to_dir(where='result') + 'Non_LCP_random_CH{}_normalized.png'.format(ch)
    plt.savefig(fig_save_filename)

    plt.close('all')
    # plt.show()

# print('Reading --> ', dataset_non_lcp_filename)
# non_lcp_df = pd.read_csv(dataset_non_lcp_filename)
# print(non_lcp_df.head())
