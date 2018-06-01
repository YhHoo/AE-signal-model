import numpy as np
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from dataset_experiment_16_5_2018 import AccousticEmissionDataSet_16_5_2018
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import optimizers
from keras.datasets import mnist
from ideal_dataset import noise_time_shift_dataset

# data = np.array([[[1, 2],
#                   [3, 4]],
#                  [[2, 3],
#                   [4, 5]],
#                  [[3, 4],
#                   [5, 6]],
#                  [[11, 12],
#                   [13, 14]],
#                  [[12, 13],
#                   [14, 15]],
#                  [[13, 14],
#                   [15, 16]]])
# print(data)
#
#
# idx = np.random.permutation(data.shape[0])
# print(idx)
# l = [0, 1, 2]
#
# data_shuffled = data[idx[:3]]
# print(data_shuffled)
#
# data[l] = data_shuffled
#
# print(data)


# time axis setting
fs = 1000
duration = 10  # tune this for duration
total_point = int(fs * duration)
time_axis = np.linspace(0, duration, total_point)
dataset, label = noise_time_shift_dataset(time_axis, fs=fs, verbose=True)

num_class = 3
class_split_index = np.linspace(0, 1581, num_class + 1)
print(class_split_index)

# accessing index btw each classes
for i in range(class_split_index.size - 1):
    # for class of index 0-10, this array will return [0, 1, ...9]
    entire_class_index = np.arange(class_split_index[i], class_split_index[i+1], 1)
    entire_class_index = [int(i) for i in entire_class_index]
    # shuffle the index
    entire_class_index_shuffled = np.random.permutation(entire_class_index)
    # shuffle the value of the class and store the shuffled values
    class_data_shuffled = dataset[entire_class_index_shuffled]
    # replace the original unshuffled matrix
    dataset[entire_class_index] = class_data_shuffled

print('AFTER SHUFFLED: ', dataset.shape)



