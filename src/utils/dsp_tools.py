# ------------------------------------------------------
# Process the Raw AE signals for training-ready
# The Raw signal is sampled at 5MHz, So time btw points = 2e-7 s
# ------------------------------------------------------

import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import spectrogram, correlate
from statsmodels.robust import mad
from scipy.signal import filtfilt, butter
from sklearn.preprocessing import MinMaxScaler


# FAST FOURIER TRANSFORM (FFT)
def fft_scipy(sampled_data=None, fs=1, visualize=True, vis_max_freq_range=None):
    '''
    :param sampled_data: A one dimensional data (Size = N), can be list or series
    :param fs: Sampling frequency
    :param visualize: Plot or not (Boolean)
    :param vis_max_freq_range: the maximum freq to include in visualization
    :return: amplitude and the frequency spectrum (Size = N // 2)
    '''
    # visualize freq
    if vis_max_freq_range is None:
        vis_max_freq_range = fs/2

    # Sample points and sampling frequency
    # N = sampled_data.size  # deprecated on 6/9/18
    N = len(sampled_data)

    # fft
    # print('Scipy.FFT on {} points...'.format(N), end='')
    # take only half of the FFT output because it is a reflection
    # take abs because the FFT output is complex
    # divide by N to reduce the amplitude to correct one
    # times 2 to restore the discarded reflection amplitude
    y_fft = fft(sampled_data)
    y_fft_mag = (2.0/N) * np.abs(y_fft[0: N//2])
    y_fft_phase = np.angle(y_fft[0: N//2])
    # x-axis - only half of N
    f_axis = np.linspace(0.0, fs/2, N//2)

    if visualize:

        # use sci. notation at the x-axis value
        plt.subplots_adjust(hspace=0.5)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plot only 0Hz to specified freq
        plt.xlim((0, vis_max_freq_range))

        # mag plot
        plt.subplot(211)
        plt.plot(f_axis, y_fft_mag)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        plt.title('Fast Fourier Transform')

        # phase plot
        plt.subplot(212)
        plt.plot(f_axis, y_fft_phase)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase')

        plt.show()

    return y_fft_mag, y_fft_phase, f_axis


# SPECTROGRAM
def spectrogram_scipy(sampled_data=None, fs=1, nperseg=1, noverlap=1, nfft=None, mode='psd',
                      return_plot=False, vis_max_freq_range=None, verbose=False,
                      save=False, plot_title='Default'):
    '''
    :param sampled_data: A one dimensional data (Size = N), can be list or series
    :param fs: Sampling frequency
    :param nperseg: if higher, f-res higher,  no of freq bin = nperseg/2 + 1 !!
    :param noverlap: if higher, t-res higher
    :param nfft: if set to 500, and if nperseg is 400, 100 zeros will be added to the 400 points signal to make it 500.
    this is to simply increase the segmented length so that more freq res is obtained.
    :param mode: 'psd', 'magnitude', 'angle'(deg), 'phase'(rad), 'complex'
    :param return_plot: return figure object of the spectrogram plot, if False, the returned obj
    wil become none
    :param verbose: Print out the transformed data summary
    :param save: save the spectrogram as .jpeg
    :param plot_title: title of the spectrogram to save
    :param vis_max_freq_range: the maximum freq to include in visualization
    :return: time axis, frequency band and the 2D matrix(shape[0]=freq, shape[1]=time step)
    '''
    # There is a trade-off btw resolution of frequency and time due to uncertainty principle
    # Spectrogram split input signal into segments before FFT and PSD on each seg.
    # Adjust nperseg is adjusting segment length. Higher nperseg giv more res in Freq but
    # lesser res in time domain.

    # ensure it is np array
    if isinstance(sampled_data, list):
        sampled_data = np.array(sampled_data)

    # check the visualize freq range
    if vis_max_freq_range is None:
        vis_max_freq_range = fs / 2

    # begin
    f, t, Sxx = spectrogram(sampled_data,
                            fs=fs,
                            scaling='spectrum',
                            nperseg=nperseg,
                            noverlap=noverlap,
                            nfft=nfft,
                            mode=mode)
    f_res = fs / (2 * (f.size - 1))
    t_res = (sampled_data.shape[0] / fs) / t.size

    # result summary
    if verbose:
        print('\n----------SPECTROGRAM OUTPUT---------')
        print('Time Segment....{}\nFirst 5: {}\nLast 5: {}\n'.format(t.size, t[:5], t[-5:]))
        print('Frequency Segment....{}\nFirst 5: {}\nLast 5: {}\n'.format(f.size, f[:5], f[-5:]))
        print('Spectrogram Dim: {}\nF-Resolution: {}Hz/Band\nT-Resolution: {}'.format(Sxx.shape, f_res, t_res))

    # plotting spectrogram
    if return_plot:
        fig = plt.figure(figsize=(8, 6))  # (x len, y len)
        fig.suptitle(plot_title)
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
        colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
        i = ax.pcolormesh(t, f, Sxx)
        fig.colorbar(i, cax=colorbar_ax)
        ax.grid()
        ax.set_xlabel('Time [Sec]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_ylim(bottom=0, top=vis_max_freq_range, auto=True)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    else:
        fig = None

    # [THIS METHOD OF PLOT IS OBSOLETE]
    # if save or return_plot:
        # plt.pcolormesh(t, f, Sxx)
        # plt.ylabel('Frequency [Hz]')
        # # display only 0Hz to 300kHz
        # plt.ylim((0, vis_max_freq_range))
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # plt.title(save_title)
        # plt.grid()
        # plt.xlabel('Time [Sec]')
        # plt.colorbar()

        # if save:
        #     plt.savefig('result\{}.png'.format(save_title))
        #
        # if return_plot:
        #     plt.show()

        # plt.close()

    return t, f, Sxx, fig


def butter_bandpass_filtfilt(sampled_data, fs, f_hicut, f_locut, order=5):
    '''
    :param sampled_data: input
    :param fs: at wat f the data is sampled
    :param f_hicut: higher boundary of the passband
    :param f_locut: lower boundary of the passband
    :param order: the higher the order the higher the Q
    :return: np array
    '''
    f_nyquist = fs / 2
    low = f_locut / f_nyquist
    high = f_hicut / f_nyquist
    b, a = butter(order, [low, high], btype='band')  # ignore warning
    # using zero phase filter (so no phase shift after filter)
    filtered_signal = filtfilt(b, a, sampled_data)

    return filtered_signal


def one_dim_xcor_2d_input(input_mat, pair_list, verbose=False):
    '''
    XCOR every band --> overall normalize
    The scipy.signal.correlate is used. Here the algorithm will starts sliding the 2 signals by bumping
    input_mat[0]'s head into the input_mat[1]'s tail. E.G. If input_mat[0] is faster, max correlation occurs at
    left side of the centre points.
    :param input_mat: a 3d np matrix input, where shape[0] -> no. of phase map (diff sensors),
                                                  shape[1] -> freq band,
                                                  shape[2] -> time steps
    :param pair_list: list of 2d tuples, e.g. [(0, 1), (1, 2)], such that input_mat[0] and input_mat[1] is xcor, and
                      input_mat[1] and input_mat[2] is xcor.
    :param verbose: print the output xcor map dimension
    :return: 3d normalized xcor map whr shape[0] -> no. of xcor maps,
                                        shape[1] -> freq band,
                                        shape[2] -> xcor steps
    '''
    # ensure they hv equal number of axis[1] or freq band
    try:
        # simply accessing
        input_mat.shape[1]
    except IndexError:
        print('YH_WARNING: The axis[1] of the input_mat are not uniform')
        raise

    xcor_bank = []
    for pair in pair_list:
        xcor_of_each_f_list = []

        # for all feature(freq/wavelet width) bands
        for i in range(input_mat.shape[1]):
            x_cor = correlate(input_mat[pair[0], i], input_mat[pair[1], i], 'full', method='fft')
            xcor_of_each_f_list.append(x_cor)

        # xcor map of 2 phase map, axis[0] is freq, axis[1] is x-cor unit shift
        xcor_of_each_f_list = np.array(xcor_of_each_f_list)

        # normalize each xcor_map with linear function btw their max and min values
        scaler = MinMaxScaler(feature_range=(0, 1))
        xcor_of_each_f_list = scaler.fit_transform(xcor_of_each_f_list.ravel().reshape((-1, 1))) \
            .reshape((xcor_of_each_f_list.shape[0], xcor_of_each_f_list.shape[1]))

        # store the xcor map for a pair of phase map
        xcor_bank.append(xcor_of_each_f_list)
    xcor_bank = np.array(xcor_bank)

    # xcor axis
    xcor_len = xcor_bank.shape[2]
    xcor_axis = np.arange(1, xcor_len + 1, 1) - xcor_len // 2 - 1

    if verbose:
        print('\n---------One-Dimensional X-correlation---------')
        print('Xcor Map Dim (No. of xcor map, freq band, xcor steps): ', xcor_bank.shape)

    return xcor_bank, xcor_axis


def one_dim_xcor_1d_input(input_mat, pair_list, verbose=False):
    '''
    This is same as the one_dim_xcor_2d_input(), except here takes only 1 frequency band, and no normalize of the
    xcor result.

    :param input_mat: a 2d np matrix input, where shape[0] -> no. of phase map (diff sensors),
                                                  shape[1] -> time steps
    :param pair_list: list of 2d tuples, e.g. [(0, 1), (1, 2)], such that input_mat[0] and input_mat[1] is xcor, and
                      input_mat[1] and input_mat[2] is xcor.
    :param verbose: print the output xcor map dimension
    :return: 3d normalized xcor map whr shape[0] -> no. of xcor maps,
                                        shape[1] -> xcor steps
    '''

    xcor_bank = []
    for pair in pair_list:
        x_cor = correlate(input_mat[pair[0]], input_mat[pair[1]], 'full', method='fft')
        # store the xcor map for a pair of phase map
        xcor_bank.append(x_cor[0])
    xcor_bank = np.array(xcor_bank)

    # xcor axis
    xcor_len = xcor_bank.shape[1]
    xcor_axis = np.arange(1, xcor_len + 1, 1) - xcor_len // 2 - 1

    if verbose:
        print('\n---------One-Dimensional X-correlation---------')
        print('Xcor Map Dim (No. of xcor map, freq band, xcor steps): ', xcor_bank.shape)

    return xcor_bank, xcor_axis


def dwt_smoothing(x, wavelet="db4", level=1):
    '''
    :param x: 1d array / list of points (1d signal)
    :param wavelet: wavelet of pywt
    :param level: the approximate level that is taken to calc the universal threshold for threshold-ing
    :return: 1d array
    '''
    # calculate the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")

    # calculate a threshold (Median Absolute Deviation) - find deviation of every item from median in a 1d array, then
    # find the median of the deviation again.
    sigma = mad(coeff[-level])

    # changing this threshold also changes the behavior. This is universal threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))

    # thresholding on all the detail components, using universal threshold from the level (-level)
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])

    # reconstruct the signal using the thresholded coefficients
    x_smoothed = pywt.waverec(coeff, wavelet, mode="per")

    return x_smoothed


def detect_ae_event_by_sandwich_sensor(x1, x2, threshold1, threshold2):
    '''
    Recommended usage:
    It is usually to process the peaks list by peakutils.indexes() from 2 sensors at SAME dist away fr AE source.

    For items in x1 and x2, if any items of x1 and x2 are closer than thre1, it will store the x2's value.
    e.g. if x1 has 34, and x2 has 35, it will only keep index from x2, which is 35. After that, if the items in
    resulting list are still too close by thre2, it will remove the smaller index.

    example --
    x1 = [1, 13, 18, 20, 34]
    x2 = [0, 15, 17, 50]
    thre1 = 2
    thre2 = 3
    first, it will produce [0, 15, 17]  --> closely similiar values
    then, it will produce  [0, 15]      --> remove the index that are closely similar within itself

    :param x1: a 1d array of list
    :param x2: a 1d array of list
    :param threshold1: for x1 and x2 similar point selection
    :param threshold2: for too-close-point removal
    :return: a list
    '''

    item_to_stay = []

    # finding closely similiar points btw x1 and x2
    for p1 in x1:
        for p2 in x2:
            abs_diff = np.abs(p2 - p1)
            if abs_diff < threshold1:
                item_to_stay.append(p2)
                break

    # for the resulting list of items from above, remove the items are too close (differ within threshold2).
    item_to_del = []
    # note down either one of the 2 items tat too close in a delete list
    for i in range(len(item_to_stay) - 1):
        if np.abs(item_to_stay[i + 1] - item_to_stay[i]) < threshold2:
            item_to_del.append(item_to_stay[i])

    # remove items according to the delete list
    for d in item_to_del:
        item_to_stay.remove(d)

    return item_to_stay


def detect_ae_event_by_v_sensor(x1, x2, x3, x4, n_ch_signal, threshold_list=None, threshold_x=None):
    '''
    This function is specific for processing 4 lists of peak indexes from peakutils.indexes()

    Algorithm:
    It is expecting [x1, x2, x3, x4] as [-3m, -2m, 2m, 4m], and the ae event
    is from 0m.
    First, it finds similiar peaks from -2m and 2m --> abs([-2m]-[2m]) < threshold[0]
    Then, it takes the midpoints of the similiar peak from -2m and 2m as mid.
    It then finds points from x1 that are > mid and has distance less than threshold[1]. if yes, set flag_0 = True
    Last, it finds points from x4 that are > mid and has distance less than threshold[2]. if yes, set flag_1 = True
    For mid with both flag_0 and flag_1 on, it is stored and returned.

    :param x1: a list of peaks indexs
    :param x2: a list of peaks indexs
    :param x3: a list of peaks indexs
    :param x4: a list of peaks indexs
    :param threshold_list: a list of 3 integers
    :param threshold_x: a single value
    :return: a list of indexes where AE event is detected
    '''

    ae_peak, ae_peak_confirmed = [], []
    ae_peak_lo, ae_peak_hi = [], []
    flag_0, flag_1 = False, False

    # Stage 1 ----------------------------------------------------------------------------------------------------------
    # finding [0m]-emitted peak btw x1 and x2
    # we allow a very small dist difference btw the 2 peaks
    for p2 in x2:
        for p3 in x3:
            abs_diff = np.abs(p3 - p2)
            if abs_diff < threshold_list[0]:
                # find the mid point
                ae_peak.append(np.mean((p2, p3)))
                ae_peak_lo.append(p2)
                ae_peak_hi.append(p3)
                break

    # Stage 2 ----------------------------------------------------------------------------------------------------------
    # for all mid detected
    for mid, mid_lo, mid_hi in zip(ae_peak, ae_peak_lo, ae_peak_hi):
        # check mid with x1
        for p1 in x1:
            if p1 > mid:
                # satisfying distance and amplitude criteria
                if ((p1-mid) < threshold_list[1]) and (n_ch_signal[1, mid_lo] > n_ch_signal[0, p1]):
                    flag_0 = True
                    # exit for loop once a match is found
                    break
        # check mid with x4
        for p4 in x4:
            if p4 > mid:
                # satisfying distance and amplitude criteria
                if (p4-mid) < threshold_list[2] and (n_ch_signal[2, mid_hi] > n_ch_signal[0, p4]):
                    flag_1 = True
                    # exit for loop once a match is found
                    break

        # verify the mid with flag 1 and 2
        if flag_0 and flag_1:
            ae_peak_confirmed.append(int(mid))

        # reset flag to 0
        flag_0, flag_1 = False, False

    # Stage 3 ----------------------------------------------------------------------------------------------------------
    # remove the peak are too close (differ within threshold_x).
    item_to_del = []
    # note down either one of the 2 items tat too close in a delete list
    for i in range(len(ae_peak_confirmed) - 1):
        if np.abs(ae_peak_confirmed[i + 1] - ae_peak_confirmed[i]) < threshold_x:
            item_to_del.append(ae_peak_confirmed[i])

    # remove items according to the delete list
    for d in item_to_del:
        ae_peak_confirmed.remove(d)

    return ae_peak_confirmed

