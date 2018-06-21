import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# import mayavi.mlab as mlb


t = np.linspace(-1, 1, 1000, endpoint=False)
offset = [0, -0.2, -0.4]
for off in offset:
    pulse1 = signal.gausspulse(t+off, fc=50)
    pulse2 = signal.gausspulse(t-0.1+off, fc=50)
    cor = np.correlate(pulse1, pulse2, 'full')
    print('USING NUMPY CORRELATE---------------')
    print(cor)
    print(cor.shape)
    print(np.argmax(cor))
    plt.title('Xcor Map')
    plt.plot(cor)
    plt.show()
    plt.close()

pulse1 = signal.gausspulse(t, fc=50)
pulse2 = signal.gausspulse(t-0.1, fc=50)
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax2 = fig1.add_subplot(2, 1, 2)
ax1.set_title('Original Pulse')
ax2.set_title('Delayed Pulse')
ax1.plot(t, pulse1)
ax2.plot(t, pulse2)
plt.subplots_adjust(hspace=0.4)
plt.show()


# Wavelet transform
# widths = np.arange(1, 31)
# cwtmatr = signal.cwt(sig, signal.ricker, widths)
# print(cwtmatr)
# print(cwtmatr.shape)


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


