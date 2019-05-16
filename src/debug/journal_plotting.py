from src.utils.helpers import *


# PLOT TEST ACC ON SEEN AND UNSEEN DATASETS ----------------------------------------------------------------------------
unseen_test_acc = [0.6326, 0.6735, 0.9955, 0.9060, 0.9025, 0.74, 0.7011]
seen_test_acc = [0.8239, 0.81554, 0.9803, 0.9881, 0.9979, 0.9871, 0.9936]
overall_acc = [0.7282, 0.7444, 0.9879, 0.9471, 0.9502, 0.8635, 0.8474]
conv_layer_no = [4, 4, 4, 5, 4, 3, 3]
exe_time_per_sample = [0.164, 0.186, 0.185, 0.185, 0.178, 0.667, 1.000]
freq = [1, 2, 25, 50, 100, 200, 1000]

heatmap = np.array(exe_time_per_sample*10).reshape((10, 7))
im = plt.imshow(heatmap, interpolation='None', cmap='Oranges', origin='lower', aspect='auto')
plt.plot(seen_test_acc, linestyle='-.', marker='x', label='Test Accuracy (SEEN)', color='black')
plt.plot(unseen_test_acc, linestyle='--', marker='x', label='Test Accuracy (UNSEEN)', color='black')
plt.plot(overall_acc, linestyle='-', marker='x', label='Overall Test Accuracy', color='black')
# plt.plot(conv_layer_no, linestyle='solid', marker='x', label='No. of Convolutional Layers')
# plt.plot(exe_time_per_sample, linestyle='dashed', marker='x', label='Exe time per sample (ms)')
cbar = plt.colorbar(im)
cbar.set_label('Execution Time Per Sample (ms)')
# markerline, stemlines, baseline = plt.stem([0, 1, 2, 3, 4, 5, 6], exe_time_per_sample, '-.')
# plt.setp(baseline, color='k', linewidth=2)
# for f, t in zip([0, 1, 2, 3, 4, 5, 6], exe_time_per_sample):
#     plt.text(f, t, t, horizontalalignment='right', verticalalignment='top', fontsize=8)

plt.title('Test Accuracy vs Training Data Sampling Rate')
plt.xlabel('Sampling Frequency (kHz)')
plt.ylabel('Accuracy')
plt.xticks([0, 1, 2, 3, 4, 5, 6], freq)
plt.grid('on', alpha=0.5)
plt.legend()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
