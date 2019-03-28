import time
from src.utils.helpers import *
from src.utils.dsp_tools import *

# # No Normalized & Downsampled to fs=25kHz (2000 points per sample) --------------------------------------------
# leak_p1 = 'G:/Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_leak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds4.csv'
# leak_p2 = 'G:/Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_leak_random_1.5bar_[0]_ds4.csv'
# leak_p3 = 'G:/Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_leak_random_1.5bar_[-3,5,7,16,17]_ds4.csv'
#
# noleak_p1 = 'G:/Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_noleak_random_1.5bar_[-4,-2,2,4,6,8,10]_ds4.csv'
# noleak_p2 = 'G:/Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_noleak_random_1.5bar_[0]_ds4.csv'
# noleak_p3 = 'G:/Experiment_3_1_2019/leak_noleak_preprocessed_dataset/dataset_noleak_random_1.5bar_[-3,5,7,16,17]_ds4.csv'

# df_leak_rand_p1 = pd.read_csv(leak_p1)
# df_leak_rand_p2 = pd.read_csv(leak_p2)
# df_leak_rand_p3 = pd.read_csv(leak_p3)
# df_noleak_rand_p1 = pd.read_csv(noleak_p1)
# df_noleak_rand_p2 = pd.read_csv(noleak_p2)
# df_noleak_rand_p3 = pd.read_csv(noleak_p3)
#
#
# # NOLEAK ---------------------------------------------------------------------------------------------------------------
# # [-4,-2,2,4,6,8,10]
# noleak_data_p1 = df_noleak_rand_p1.loc[df_noleak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
# # shuffle
# noleak_data_p1 = noleak_data_p1[np.random.permutation(len(noleak_data_p1))]
#
# # [0]
# noleak_data_p2 = df_noleak_rand_p2.values[:, :-1]
# # shuffle
# noleak_data_p2 = noleak_data_p2[np.random.permutation(len(noleak_data_p2))]
#
# # [-3,5,7,16,17]
# noleak_data_p3 = df_noleak_rand_p3.values[:, :-1]
# # shuffle
# noleak_data_p3 = noleak_data_p3[np.random.permutation(len(noleak_data_p3))]
#
# # link all
# noleak_all = np.concatenate((noleak_data_p1[:50, :], noleak_data_p2[:50, :], noleak_data_p3[:50, :]), axis=0)
#
# # LEAK -----------------------------------------------------------------------------------------------------------------
# # [-4,-2,2,4,6,8,10]
# leak_data_p1 = df_leak_rand_p1.loc[df_leak_rand_p1['channel'].isin([2, 3, 4, 5, 6])].values[:, :-1]
# # shuffle
# leak_data_p1 = leak_data_p1[np.random.permutation(len(leak_data_p1))]
#
# # [0]
# leak_data_p2 = df_leak_rand_p2.values[:, :-1]
# # shuffle
# leak_data_p2 = leak_data_p2[np.random.permutation(len(leak_data_p2))]
#
# # [-3,5,7,16,17]
# leak_data_p3 = df_leak_rand_p3.values[:, :-1]
# # shuffle
# leak_data_p3 = leak_data_p3[np.random.permutation(len(leak_data_p3))]
#
# # link all
# leak_all = np.concatenate((leak_data_p1[:50, :], leak_data_p2[:50, :], leak_data_p3[:50, :]), axis=0)
# print('TOTAL NOLEAK:', noleak_all.shape)
# print('TOTAL LEAK:', leak_all.shape)
#
# # choose only 2 from leak and no leak
# all = np.concatenate((noleak_all[:2], leak_all[:2]), axis=0)

data_leak_noleak_filename = 'E:/Experiment_3_1_2019/leak_noleak_preprocessed_dataset/ds4_extract.csv'
data_leak_noleak = pd.read_csv(data_leak_noleak_filename).values[:, 1:]

lcp_model = load_model(model_name='LNL_29x1')
print(lcp_model.summary())
lcp_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
# activation = get_activations(lcp_model,
#                              model_inputs=data_leak_noleak.reshape((200, 2000, 1)),
#                              print_shape_only=True)

# FOR CONV KERNELS -----------------------------------------------------------------------------------------------------
# conv_1_act = activation[18]  # ** dim: (2, 2000, 32),  conv[3, 8, 13, 18]
# conv_1_act = np.swapaxes(conv_1_act, 1, 2)  # **dim: (2, 32, 2000)
# conv_layer_no = 4  # **
#
#
# for kernel_no_to_visualize in range(128):  # **
#     # plotting
#     fig = plt.figure(figsize=(12, 5))
#     fig.suptitle('conv{}_activation, kernel_no:{}'.format(conv_layer_no, kernel_no_to_visualize), fontweight="bold", size=8)
#     fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.07)
#
#     # before conv
#     ax1 = fig.add_subplot(2, 1, 1)
#     ax1.set_title('Input to Conv1')
#     ax1.plot(data_leak_noleak[0], color='red', label='Leak', alpha=0.5)
#     # ax1.plot(all[1], color='blue', label='No Leak', alpha=0.5)
#     ax1.plot(data_leak_noleak[1], color='blue', label='No Leak')
#     # ax1.plot(all[3], color='red', label='Leak')
#     handles, labels = ax1.get_legend_handles_labels()
#     display = (0, 1)
#     ax1.legend([handle for i, handle in enumerate(handles) if i in display],
#                [label for i, label in enumerate(labels) if i in display], loc='best')
#
#     # first plot
#     ax2 = fig.add_subplot(2, 1, 2)
#     ax2.set_title('After Conv{}'.format(conv_layer_no))
#     ax2.plot(conv_1_act[0, kernel_no_to_visualize, :], color='red', label='Leak', alpha=0.5)
#     # ax2.plot(conv_1_act[1, kernel_no_to_visualize, :], color='blue', label='No Leak', alpha=0.5)
#     ax2.plot(conv_1_act[1, kernel_no_to_visualize, :], color='blue', label='No Leak')
#     # ax2.plot(conv_1_act[3, kernel_no_to_visualize, :], color='red', label='Leak')
#     handles, labels = ax2.get_legend_handles_labels()
#     display = (0, 1)
#     ax2.legend([handle for i, handle in enumerate(handles) if i in display],
#                [label for i, label in enumerate(labels) if i in display], loc='best')
#
#     # plt.show()
#
#     fig_save_filename = direct_to_dir(where='result') + 'conv{}_act_kernel_{}.png'.format(conv_layer_no,
#                                                                                           kernel_no_to_visualize)
#     fig.savefig(fname=fig_save_filename)
#
#     plt.close('all')

# FOR GAP --------------------------------------------------------------------------------------------------------------
# gap_act = activation[22]  # **
# fig = plt.figure(figsize=(6, 5))
# fig.suptitle('Global Average Pooling', fontweight="bold", size=8)
# ax1 = fig.add_subplot(1, 1, 1)
#
# for i in range(0, 100, 1):
#     ax1.plot(gap_act[i], color='red', label='Leak',  marker='x', linestyle='None')
# for j in range(100, 200, 1):
#     ax1.plot(gap_act[j], color='blue', label='No Leak', marker='x', linestyle='None', alpha=0.5)
#
#
# handles, labels = ax1.get_legend_handles_labels()
# display = (0, 100)
# ax1.legend([handle for i, handle in enumerate(handles) if i in display],
#            [label for i, label in enumerate(labels) if i in display], loc='best')
#
# plt.show()





