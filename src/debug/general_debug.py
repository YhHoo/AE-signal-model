import numpy as np
import sys
from src.utils.helpers import read_all_tdms_from_folder, three_dim_visualizer


# three_dim_visualizer()
data_3d = np.array([[[1],
                     [3]],
                    [[2],
                     [4]],
                    [[3],
                     [5, 5]]])
# print(data_3d.shape[2])
try:
    no_of_freq_band = data_3d.shape[2]
    pass
except IndexError:
    print('YH_WARNING: The axis[1] of the input_mat are not equal')
    raise



print('COMPLETED')

# print(data_3d.shape[1])
# print(data_3d[0].shape[0])



# data_2d = np.array([[1, 2],
#                     [3, 4],
#                     [2, 5]])
# data_2d = data_2d.astype(dtype='float32')

# mat1 = np.array([1, 0, 3, 3, 2, 1, 0, 0]).reshape((1, 8))
# mat2 = np.array([3, 1, 0, 3, 3, 2, 1, 0]).reshape((1, 8))
#
# mat3 = np.array([0, 0, 1, 0, 1, 0, 0, 0]).reshape((1, 8))
# mat4 = np.array([0, 0, 0, 1, 0, 1, 0, 0]).reshape((1, 8))
#
# ori_signal = np.concatenate((mat1, mat3), axis=0)
# lag_signal = np.concatenate((mat2, mat4), axis=0)
# print(ori_signal.shape)

# cor = np.correlate(mat1, mat2, 'full')
# print('USING NUMPY CORRELATE---------------')
# print(cor)
# print(cor.shape)
# print(np.argmax(cor))

# cor = correlate(ori_signal, lag_signal, 'full')
# print('USING SCIPY CORRELATE---------------')
# print(cor)
# print(cor.shape)
# print(np.argmax(cor))

