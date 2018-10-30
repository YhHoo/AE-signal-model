'''
THIS SCRIPT IS FOR XCOR ALL PLB IN ALL CHANNELS, GET THEIR XCOR INDEX AND PROMINENT FREQUENCY USING CROSS SPECTRUM
DEN
'''
from scipy.interpolate import interp1d
from scipy.signal import correlate
from src.utils.helpers import *


# LOCATING XCOR INDEX AND PROMINENT F ----------------------------------------------------------------------------------

# plb_near_dir = 'E:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/PLB/0m/'
# plb_far_dir = 'E:/Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/PLB/0m/'
#
# dominant_f_save_filename = direct_to_dir(where='result') + 'dominant_f_far_(-).csv'
# xcor_index_save_filename = direct_to_dir(where='result') + 'xcor_index_far_(-).csv'
#
# # xcor pairing commands - [near] = 0m, 1m,..., 10m
# sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
# sensor_pair_near_inv = [(pair[1], pair[0]) for pair in sensor_pair_near]
# # xcor pairing commands - [far] = 11m, 12m,..., 20m
# sensor_pair_far = [(0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
# sensor_pair_far_inv = [(pair[1], pair[0]) for pair in sensor_pair_far]
#
# all_file_path = [(plb_far_dir + f) for f in listdir(plb_far_dir) if f.endswith('.tdms')]
#
# all_dominant_f = {}
# all_xcor_index = {}
# for i in range(11, 21, 1):
#     all_dominant_f['class_[{}]'.format(i)] = []
#     all_xcor_index['class_[{}]'.format(i)] = []
#
# for f in all_file_path:
#     foi = read_single_tdms(filename=f)
#     foi = np.swapaxes(foi, 0, 1)[:, 90000:130000]
#
#     xcor_all = []
#     for pair, dist_diff in zip(sensor_pair_far_inv, np.arange(11, 21, 1)):
#         xcor_one_dim = correlate(in1=foi[pair[0]], in2=foi[pair[1]], mode='full', method='fft')
#         xcor_all.append(xcor_one_dim)
#         max_xcor_index = np.argmax(xcor_one_dim)
#
#         xcor_relative = int(max_xcor_index - len(xcor_one_dim)/2)
#         print('XCOR relative: ', xcor_relative)
#
#         all_xcor_index['class_[{}]'.format(dist_diff)].append(xcor_relative)
#
#     for xcor, dist_diff in zip(xcor_all, np.arange(11, 21, 1)):
#         amp, _, freq_axis = fft_scipy(sampled_data=xcor, fs=1e6, visualize=False)
#
#         max_amp_index = np.argmax(amp)
#         print('Dist Diff[{}m] Max Freq @ {:.4f}Hz'.format(dist_diff, freq_axis[max_amp_index]))
#
#         all_dominant_f['class_[{}]'.format(dist_diff)].append(freq_axis[max_amp_index])
#
#     # print(all_dominant_f)
#     # print(all_xcor_index)
#
#
# df = pd.DataFrame.from_dict(all_dominant_f)
# print(df)
# df.to_csv(dominant_f_save_filename)
# print('saving --> ', dominant_f_save_filename)
#
# df2 = pd.DataFrame.from_dict(all_xcor_index)
# print(df2)
# df2.to_csv(xcor_index_save_filename)
# print('saving --> ', xcor_index_save_filename)


# FIND LOCATION --------------------------------------------------------------------------------------------------------

wave_speed_filename = direct_to_dir(where='desktop') + 'F11_vdisp.csv'
dominant_f_filename = 'C:/Users/YH/Desktop/hooyuheng.masterWork/' + 'dominant_f_far_(-).csv'
xcor_index_filename = 'C:/Users/YH/Desktop/hooyuheng.masterWork/' + 'xcor_index_far_(-).csv'

df_wavespeed = pd.read_csv(wave_speed_filename)
df_dominant_f = pd.read_csv(dominant_f_filename)
df_xcor_index = pd.read_csv(xcor_index_filename)

x = df_wavespeed['frequency'].values
y = df_wavespeed['wavespeed'].values

f_wavespeed = interp1d(x=x, y=y, kind='cubic')

for dist_diff in range(11, 21, 1):
    print('DIST DIFF OF {}m'.format(dist_diff))
    dominant_f = df_dominant_f['class_[{}]'.format(dist_diff)].values
    xcor_index = df_xcor_index['class_[{}]'.format(dist_diff)].values

    error_all = []
    for f, x in zip(dominant_f, xcor_index):
        dist_diff_predicted = (f_wavespeed(f) * x) / 1e6
        # print('Predicted: {}m, Actual: {}m'.format(dist_diff_predicted, dist_diff))

        error = np.abs(np.abs(dist_diff_predicted) - dist_diff)
        error_all.append(error)

    print('Mean Error: {:.2f}m'.format(np.mean(error_all)))






