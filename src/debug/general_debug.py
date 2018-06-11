import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# import mayavi.mlab as mlb


# three_dim_visualizer()
data_3d = np.array([[[1],
                     [3]],
                    [[2],
                     [4]],
                    [[3],
                     [5]]])
print(data_3d.shape)
# print(data_3d.shape[1])
# print(data_3d[0].shape[0])

# s = np.arange(0, 100, 1).reshape((5, 20))
# x = np.arange(0, 20, 1)
# y = np.arange(0, 5, 1)
# print(s)
# print(x)
# print(y)
# mlb.barchart(y, x, s)
# mlb.imshow()
sig = np.repeat([0., 1., 0., 1], 100)
win = signal.hann(50)
print(sum(win))
mat1 = np.array([3, 9, 2, 1, 0, 0, 0, 0])
mat2 = np.array([0, 3, 9, 2, 1, 0, 0, 0])

# mat3 = np.array([0, 0, 1, 0, 1, 0, 0, 0]).reshape((1, 8))
# mat4 = np.array([0, 0, 0, 1, 0, 1, 0, 0]).reshape((1, 8))
plt.plot(sig)
plt.plot(win, marker='x')
plt.show()


# ori_signal = np.concatenate((mat1, mat3), axis=0)
# lag_signal = np.concatenate((mat2, mat4), axis=0)
# print(ori_signal.shape)

# cor = np.correlate(mat2, mat1, 'full')
# print('USING NUMPY CORRELATE---------------')
# print(cor)
# print(cor.shape)
# print(np.argmax(cor))
# plt.plot(cor)
# plt.show()

