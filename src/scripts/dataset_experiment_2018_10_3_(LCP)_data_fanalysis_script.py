from src.utils.helpers import *
from src.utils.dsp_tools import *

input_data_labels_2 = ['sensor@[-3m]',
                       'sensor@[-2m]',
                       'sensor@[0m]',
                       'sensor@[5m]',
                       'sensor@[7m]',
                       'sensor@[16m]',
                       'sensor@[17m]']

correctly_classified_unseen_noleak_filename = 'E:/Experiment_3_1_2019/-3,-2,0,5,7,16,17/1.5 bar/NoLeak/' \
                                              'Test data/2019.01.03_110756_012.tdms'
wrongly_classified_unseen_noleak_filename = 'E:/Experiment_3_1_2019/-3,-2,0,5,7,16,17/1.5 bar/NoLeak/' \
                                              'Test data/2019.01.03_110741_009.tdms'

# train_data_filename = 'E:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/test1_0017.tdms'
# train_data_filename_2 = 'E:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/test1_0040.tdms'
data_1 = read_single_tdms(wrongly_classified_unseen_noleak_filename)
data_1 = np.swapaxes(data_1, 0, 1)[:-1, :]
# data_2 = read_single_tdms(wrongly_classified_unseen_noleak_filename)
# data_2 = np.swapaxes(data_2, 0, 1)[:-1, :]
# print(data_1.shape)
# print(data_2.shape)

# fig1 = plot_multiple_timeseries(input=data_1,
#                                 subplot_titles=['sensor@[-3m]',
#                                                 'sensor@[-2m]',
#                                                 'sensor@[0m]',
#                                                 'sensor@[5m]',
#                                                 'sensor@[7m]',
#                                                 'sensor@[16m]',
#                                                 'sensor@[17m]'],
#                                 main_title='correctly_classified_unseen_noleak')
#
# fig2 = plot_multiple_timeseries(input=data_2,
#                                 subplot_titles=['sensor@[-3m]',
#                                                 'sensor@[-2m]',
#                                                 'sensor@[0m]',
#                                                 'sensor@[5m]',
#                                                 'sensor@[7m]',
#                                                 'sensor@[16m]',
#                                                 'sensor@[17m]'],
#                                 main_title='wrongly_classified_unseen_noleak')

i = 0
for ch, label in zip(data_1, input_data_labels_2):
    f_mag_unseen, _, f_axis = fft_scipy(sampled_data=ch, fs=int(200e3), visualize=False)
    fig3 = plt.figure(figsize=(12, 8))
    fig3.suptitle('Correct UNSEEN NOLEAK FFT of ' + label)
    ax_fft = fig3.add_subplot(1, 1, 1)
    ax_fft.plot(f_axis[10:], f_mag_unseen[10:], alpha=0.5)
    ax_fft.set_ylim(bottom=0, top=0.001)
    ax_fft.grid('on')
    save_filename = direct_to_dir(where='result') + 'fft_{}_w.png'.format(i)
    fig3.savefig(save_filename)
    print('Saved --> ', save_filename)
    i += 1

    plt.close('all')

# f_mag_unseen, _, f_axis = fft_scipy(sampled_data=data_1, fs=int(200e3), visualize=False)
# f_mag_train, _, _ = fft_scipy(sampled_data=data_2, fs=int(200e3), visualize=False)
#
# plt.plot(f_axis[10:], f_mag_unseen[10:], color='b', alpha=0.5, label='correctly_classified_unseen_noleak')
# plt.plot(f_axis[10:], f_mag_train[10:], color='r', alpha=0.5, label='wrongly_classified_unseen_noleak')
# plt.grid('on')
# plt.legend()
# plt.show()



