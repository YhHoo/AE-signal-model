import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# import mayavi.mlab as mlb


fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(1, 1, 1)
plt.plot([1, 2, 3])

# ax.plot([1, 2, 3], marker='x')
# ax.legend()
# ax.pcolormesh
plt.show()




# def test(x, y):
#     assert x == y, 'Non equal xy'
#
#
# test(5, 6)

# three_dim_visualizer()
# data_3d = np.array([[[1],
#                      [3]],
#                     [[2],
#                      [4]],
#                     [[3],
#                      [5]]])
# print(data_3d.shape)
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

# sig = np.repeat([0., 1., 0., 1], 100)
# win = signal.hann(50)
# print(sum(win))
# mat1 = np.array([3, 9, 2, 1, 0, 0, 0, 0])
# mat2 = np.array([0, 3, 9, 2, 1, 0, 0, 0])
# plt.plot(sig)
# plt.plot(win, marker='x')
# plt.show()


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

