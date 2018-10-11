import time
import pywt
from scipy import signal
from src.utils.dsp_tools import spectrogram_scipy
from src.utils.helpers import *


filename = 'lcp_index_1bar_near_segmentation4'
lcp_index_dir = 'C:/Users/YH/Desktop/hooyuheng.masterWork/LCP DATASET OCT 3 1BAR/'

all_file_path = [(lcp_index_dir + f) for f in listdir(lcp_index_dir) if f.endswith('.csv')]


lcp_count = 0

all_ch_count = {}
for i in range(6):
    all_ch_count['ch{}'.format(i)] = 0

for f in all_file_path:
    print(f)
    df = pd.read_csv(f, index_col=0)

    # total LCP
    data_ravel = df.values[:, 2:].ravel()
    lcp_count += np.count_nonzero(data_ravel)

    # total channel
    for i in range(6):
        ch_row = df.values[:, i+2].ravel()
        all_ch_count['ch{}'.format(i)] += np.count_nonzero(ch_row)

print('LCP accumulate: ', lcp_count)
print(all_ch_count)



