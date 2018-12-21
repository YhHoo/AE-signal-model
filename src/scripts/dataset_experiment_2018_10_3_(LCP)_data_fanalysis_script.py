from src.utils.helpers import *
from src.utils.dsp_tools import *


unseen_data_filename = 'E:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/2 bar/No_Leak/test_0017.tdms'
train_data_filename = 'E:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/test1_0017.tdms'
train_data_filename_2 = 'E:/Experiment_2_10_2018/-4.5,-2,2,5,8,17,20,23/no_leak/test1_0040.tdms'
unseen_data = read_single_tdms(unseen_data_filename)
unseen_data = np.swapaxes(unseen_data, 0, 1)
train_data = read_single_tdms(train_data_filename)
train_data = np.swapaxes(train_data, 0, 1)

# normalize
scaler = MinMaxScaler(feature_range=(-1, 1))
signal_1 = scaler.fit_transform(unseen_data[1].reshape(-1, 1)).ravel()
signal_2 = scaler.fit_transform(train_data[1].reshape(-1, 1)).ravel()

f_mag_unseen, _, f_axis = fft_scipy(sampled_data=signal_1, fs=int(1e6), visualize=False)
f_mag_train, _, _ = fft_scipy(sampled_data=signal_2, fs=int(1e6), visualize=False)

plt.plot(f_axis[10:], f_mag_unseen[10:], color='b', alpha=0.5, label='signal 1')
plt.plot(f_axis[10:], f_mag_train[10:], color='r', alpha=0.5, label='signal 2')
plt.grid('on')
plt.legend()
plt.show()



