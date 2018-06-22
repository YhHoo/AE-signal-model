import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from src.controlled_dataset.ideal_dataset import white_noise
# import mayavi.mlab as mlb

t = np.linspace(0, 10, 11)
f = np.linspace(10, 100, 11)
mat = np.arange(0, 100, 1).reshape((10, 10))
print(t.shape)
print(f.shape)
print(mat.shape)
print(mat)


def colormapplot():
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
    i = ax.pcolormesh(t, f, mat)
    ax.grid()
    fig.colorbar(i, cax=colorbar_ax)

    return fig


for i in range(3):
    _ = colormapplot()
    plt.close()

fig1 = colormapplot()
plt.show()




# three_dim_visualizer()
# data_3d = np.array([[[1],
#                      [3]],
#                     [[2],
#                      [4]],
#                     [[3],
#                      [-5]]])
# print(data_3d.min())
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


