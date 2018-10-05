import time
import pywt
from scipy import signal
from src.utils.dsp_tools import spectrogram_scipy
from src.utils.helpers import *


# channel naming
label = ['-4.5m', '-2m', '2m', '5m', '8m', '17m', '20m', '23m']

tdms_dir = 'E:/Experiment_3_10_2018/-4.5, -2, 2, 5, 8, 10, 17 (leak 1bar)/'

all_tdms_file = [(tdms_dir + f) for f in listdir(tdms_dir) if f.endswith('.tdms')]

n_channel_data_near_leak = read_single_tdms(all_tdms_file[0])
n_channel_data_near_leak = np.swapaxes(n_channel_data_near_leak, 0, 1)


# peak_list = []
# time_start = time.time()
# for channel in n_channel_data_near_leak:
#     print('Peak Detecting')
#     peak, _ = signal.find_peaks(x=channel, distance=3000, prominence=(0.5, None))
#     peak_list.append(peak)
# print('Time Taken for peakutils.indexes(): {:.4f}s'.format(time.time()-time_start))
#
#
# fig_timeseries = plot_multiple_timeseries_with_roi(input=n_channel_data_near_leak,
#                                                    subplot_titles=label,
#                                                    main_title='No Leak',
#                                                    all_ch_peak=peak_list,
#                                                    lcp_list=[])
#
# plt.show()

soi = n_channel_data_near_leak[1, 3737370:(3737370+6000)]

soi_cwt, _ = pywt.cwt(soi, scales=np.linspace(1, 50, 1000), wavelet='gaus1')
fig_1 = plot_cwt_with_time_series(time_series=soi,
                                  no_of_time_series=1,
                                  cwt_mat=soi_cwt,
                                  cwt_scale=np.linspace(1, 50, 1000),
                                  title='CWT of LCP [17m]')


plt.show()




