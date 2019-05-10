from src.utils.helpers import *


# PLOT TEST ACC ON SEEN AND UNSEEN DATASETS ----------------------------------------------------------------------------
unseen_test_acc = [0.6326, 0.6735, 0.9955, 0.9060, 0.9025, 0.74, 0.7011]
seen_test_acc = [0.8239, 0.81554, 0.9803, 0.9881, 0.9979, 0.9871, 0.9936]
freq = [1, 2, 25, 50, 100, 200, 1000]

plt.plot(seen_test_acc, linestyle='solid', marker='x', label='Test Accuracy (SEEN)')
plt.plot(unseen_test_acc, linestyle='dashed', marker='x', label='Test Accuracy (UNSEEN)')
plt.xlabel('Sampling Frequency (kHz)')
plt.ylabel('Accuracy')
plt.xticks([0, 1, 2, 3, 4, 5, 6], freq)
plt.grid('on', alpha=0.5)
plt.legend()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
