import time
from src.utils.helpers import *
from src.utils.dsp_tools import *

# No Normalized & Downsampled to fs=25kHz (2000 points per sample) --------------------------------------------
leak_p1 = 'G:/Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds4.csv'
leak_p2 = 'G:/Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_leak_random_1.5bar_[0]_ds4.csv'
leak_p3 = 'G:/Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_leak_random_1.5bar_[-3,5,7,16,17]_ds4.csv'

df_leak_rand_p1 = pd.read_csv(leak_p1)
df_leak_rand_p2 = pd.read_csv(leak_p2)


# LEAK -----------------------------------------------------------------------------------------------------------------
leak0m = df_leak_rand_p2.values[50:70, :-1].reshape((20, 2000))

leak2m = df_leak_rand_p1.loc[df_leak_rand_p1['channel'] == 2].values[50:70, :-1].reshape((20, 2000))

leak4m = df_leak_rand_p1.loc[df_leak_rand_p1['channel'] == 3].values[50:70, :-1].reshape((20, 2000))

leak6m = df_leak_rand_p1.loc[df_leak_rand_p1['channel'] == 4].values[50:70, :-1].reshape((20, 2000))

leak8m = df_leak_rand_p1.loc[df_leak_rand_p1['channel'] == 5].values[50:70, :-1].reshape((20, 2000))

leak10m = df_leak_rand_p1.loc[df_leak_rand_p1['channel'] == 6].values[50:70, :-1].reshape((20, 2000))


# link all
leak_all = np.concatenate((leak0m, leak2m, leak4m, leak6m, leak8m, leak10m), axis=0)

print(leak_all.shape)

lcp_model = load_model(model_name='LD_10x4')
print(lcp_model.summary())
lcp_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
activation = get_activations(lcp_model,
                             model_inputs=leak_all.reshape((120, 2000, 1)),
                             print_shape_only=True)

# conv_act = activation[26]  # dim: (6, 2000, 32)
# conv_act = np.swapaxes(conv_act, 1, 2)  # dim: (6, 32, 2000)
# conv_no = 6
#
# for kernel_no in range(conv_act.shape[1]):
#     fig = plt.figure(figsize=(12, 8))
#     fig.suptitle('conv{}_kernel{}'.format(conv_no, kernel_no), fontweight="bold", size=12)
#     ax1 = fig.add_subplot(2, 1, 1)
#     ax1.plot(leak_all[0], color='blue', label='Leak @ 0m', alpha=0.5)
#     ax1.plot(leak_all[1], color='red', label='Leak @ 2m', alpha=0.5)
#     ax1.plot(leak_all[2], color='yellow', label='Leak @ 4m', alpha=0.5)
#     ax1.plot(leak_all[3], color='green', label='Leak @ 6m', alpha=0.5)
#     ax1.plot(leak_all[4], color='cyan', label='Leak @ 8m', alpha=0.5)
#     ax1.plot(leak_all[5], color='magenta', label='Leak @ 10m', alpha=0.5)
#     ax1.legend()
#
#     ax2 = fig.add_subplot(2, 1, 2)
#     ax2.plot(conv_act[0, kernel_no], color='blue', label='Leak @ 0m', alpha=0.5)
#     ax2.plot(conv_act[1, kernel_no], color='red', label='Leak @ 2m', alpha=0.5)
#     ax2.plot(conv_act[2, kernel_no], color='yellow', label='Leak @ 4m', alpha=0.5)
#     ax2.plot(conv_act[3, kernel_no], color='green', label='Leak @ 6m', alpha=0.5)
#     ax2.plot(conv_act[4, kernel_no], color='cyan', label='Leak @ 8m', alpha=0.5)
#     ax2.plot(conv_act[5, kernel_no], color='magenta', label='Leak @ 10m', alpha=0.5)
#     ax2.legend()
#
#     fig_save_filename = direct_to_dir(where='result') + 'conv{}_kernel{}.png'.format(conv_no, kernel_no)
#     fig.savefig(fig_save_filename)
#     plt.close('all')

# # plotting
# fig = plt.figure(figsize=(12, 5))
# fig.suptitle('conv1_activation', fontweight="bold", size=8)
# fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.03)
#
# # before conv
# ax1 = fig.add_subplot(2, 1, 1)
# ax1.set_title('Input to Conv1')
# ax1.plot(all[0], color='blue', label='No Leak', alpha=0.5)
# ax1.plot(all[1], color='blue', label='No Leak', alpha=0.5)
# ax1.plot(all[2], color='red', label='Leak')
# ax1.plot(all[3], color='red', label='Leak')
# handles, labels = ax1.get_legend_handles_labels()
# display = (0, 2)
# ax1.legend([handle for i, handle in enumerate(handles) if i in display],
#            [label for i, label in enumerate(labels) if i in display], loc='best')
#
# # first plot
# ax2 = fig.add_subplot(2, 1, 2)
# ax2.set_title('After Conv1')
# ax2.plot(conv_1_act[0, kernel_no_to_visualize, :], color='blue', label='No Leak', alpha=0.5)
# ax2.plot(conv_1_act[1, kernel_no_to_visualize, :], color='blue', label='No Leak', alpha=0.5)
# ax2.plot(conv_1_act[2, kernel_no_to_visualize, :], color='red', label='Leak')
# ax2.plot(conv_1_act[3, kernel_no_to_visualize, :], color='red', label='Leak')
# handles, labels = ax2.get_legend_handles_labels()
# display = (0, 2)
# ax2.legend([handle for i, handle in enumerate(handles) if i in display],
#            [label for i, label in enumerate(labels) if i in display], loc='best')
#
# plt.show()


# FOR GAP --------------------------------------------------------------------------------------------------------------
gap_act = activation[26]  # **
fig = plt.figure(figsize=(6, 5))
fig.suptitle('Global Average Pooling', fontweight="bold", size=8)
ax1 = fig.add_subplot(1, 1, 1)
line_labels = ['Leak@0m', 'Leak@2m', 'Leak@4m', 'Leak@6m', 'Leak@8m', 'Leak@10m']

l0, l2, l4, l6, l8, l10 = [], [], [], [], [], []

for i in range(0, 20, 1):
    ax1.plot(gap_act[i], color='blue', label='Leak@0m',  marker='x', linestyle='None', alpha=0.5)

for i in range(20, 40, 1):
    ax1.plot(gap_act[i], color='red', label='Leak@2m',  marker='x', linestyle='None', alpha=0.5)

for i in range(40, 60, 1):
    ax1.plot(gap_act[i], color='yellow', label='Leak@4m',  marker='x', linestyle='None', alpha=0.5)

for i in range(60, 80, 1):
    ax1.plot(gap_act[i], color='green', label='Leak@6m',  marker='x', linestyle='None', alpha=0.5)

for i in range(80, 100, 1):
    ax1.plot(gap_act[i], color='cyan', label='Leak@8m',  marker='x', linestyle='None', alpha=0.5)

for i in range(100, 120, 1):
    ax1.plot(gap_act[i], color='magenta', label='Leak@10m',  marker='x', linestyle='None', alpha=0.5)

ax1.grid('on')
# handles, labels = ax1.get_legend_handles_labels()
# display = (0, 20)
# ax1.legend([handle for i, handle in enumerate(handles) if i in display],
#            [label for i, label in enumerate(labels) if i in display], loc='best')

plt.show()




