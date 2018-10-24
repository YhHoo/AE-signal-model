from scipy.signal import correlate
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.utils.helpers import *

plb_near_dir = 'E:/Experiment_13_7_2018/Experiment 1/-3,-2,2,4,6,8,10,12/PLB/0m/'
plb_far_dir = 'E:/Experiment_13_7_2018/Experiment 1/-3,-2,10,14,16,18,20,22/PLB/0m/'

# xcor pairing commands - [near] = 0m, 1m,..., 10m
sensor_pair_near = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]
# xcor pairing commands - [far] = 11m, 12m,..., 20m
sensor_pair_far = [(0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7)]

all_file_path = [(plb_near_dir + f) for f in listdir(plb_near_dir) if f.endswith('.tdms')]

all_class = {}
for i in range(0, 11, 1):
    all_class['class_[{}]'.format(i)] = []

for f in all_file_path:
    foi = read_single_tdms(filename=f)
    foi = np.swapaxes(foi, 0, 1)[:, 90000:130000]

    xcor_all = []
    for pair in sensor_pair_near:
        xcor_one_dim = correlate(in1=foi[pair[0]], in2=foi[pair[1]], mode='full', method='fft')
        xcor_all.append(xcor_one_dim)

    for xcor, dist_diff in zip(xcor_all, np.arange(0, 11, 1)):
        amp, _, freq_axis = fft_scipy(sampled_data=xcor, fs=1e6, visualize=False)

        max_amp_index = np.argmax(amp)
        print('Dist Diff[{}m] Max Freq @ {:.4f}Hz'.format(dist_diff, freq_axis[max_amp_index]))

        all_class['class_[{}]'.format(dist_diff)].append(freq_axis[max_amp_index])


    # plt.show()
temp = []
for i in range(0, 11, 1):
    temp.append(all_class['class_[{}]'.format(i)])

dominant_f_all = np.concatenate(temp, axis=0)
print(dominant_f_all.shape)

print(dominant_f_all)



