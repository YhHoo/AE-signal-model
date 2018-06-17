import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# import mayavi.mlab as mlb


# fig1 = plt.figure()
# fig2 = plt.figure()
# ax1 = fig1.add_subplot(1, 1, 1)
# ax2 = fig2.add_subplot(1, 1, 1)
# ax1.plot([1, 2, 3])
# ax2.plot([5, 6, 7])
# plt.show()
t = np.linspace(-1, 1, 200, endpoint=False)
sig = np.cos(2 * np.pi * 7 * t)
pulse = signal.gausspulse(t - 0.4, fc=2)
widths = np.arange(1, 31)
cwtmatr = signal.cwt(sig, signal.ricker, widths)
print(cwtmatr)
print(cwtmatr.shape)


fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title('Wavelet Transform')
ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title('Input Signal')

ax1.plot(t, sig)
ax1.set_xbound(lower=t.min(), upper=t.max())

i = ax2.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
               vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
ax2.set_xbound(lower=t.min(), upper=t.max())
# fig1.colorbar(i)
plt.subplots_adjust(hspace=0.3)
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

# cor = np.correlate(mat2, mat1, 'full')
# print('USING NUMPY CORRELATE---------------')
# print(cor)
# print(cor.shape)
# print(np.argmax(cor))
# plt.plot(cor)
# plt.show()

