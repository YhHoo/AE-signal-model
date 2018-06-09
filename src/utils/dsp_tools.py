# ------------------------------------------------------
# Process the Raw AE signals for training-ready
# The Raw signal is sampled at 5MHz, So time btw points = 2e-7 s
# ------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import spectrogram
from scipy.signal import filtfilt, butter
from sklearn.preprocessing import MinMaxScaler


# FAST FOURIER TRANSFORM (FFT)
def fft_scipy(sampled_data=None, fs=1, visualize=True, vis_max_freq_range=1e3):
    '''
    :param sampled_data: A one dimensional data (Size = N), can be list or series
    :param fs: Sampling frequency
    :param visualize: Plot or not (Boolean)
    :param vis_max_freq_range: the maximum freq to include in visualization
    :return: amplitude and the frequency spectrum (Size = N // 2)
    '''
    # Sample points and sampling frequency
    N = sampled_data.size
    # fft
    print('Scipy.FFT on {} points...'.format(N), end='')
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
    print('[Done]')

    return y_fft_mag, y_fft_phase, f_axis


# SPECTROGRAM
def spectrogram_scipy(sampled_data=None, fs=1, nperseg=1, noverlap=1, mode='psd',
                      visualize=False, vis_max_freq_range=1e3, verbose=False,
                      save=False, save_title='Default'):
    '''
    :param sampled_data: A one dimensional data (Size = N), can be list or series
    :param fs: Sampling frequency
    :param nperseg: if higher, f-res higher,  no of freq bin = nperseg/2 + 1 !!
    :param noverlap: if higher, t-res higher
    :param visualize: Plot Spectrogram or not (Boolean)
    :param verbose: Print out the transformed data summary
    :param save: save the spectrogram as .jpeg
    :param save_title: title of the spectrogram to save
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
    # begin
    f, t, Sxx = spectrogram(sampled_data,
                            fs=fs,
                            scaling='spectrum',
                            nperseg=nperseg,  # ori=10000
                            noverlap=noverlap,  # ori=5007
                            mode=mode)
    f_res = fs / (2 * (f.size - 1))
    t_res = (sampled_data.shape[0] / fs) / t.size

    # result summary
    if verbose:
        print('\n----------SPECTROGRAM OUTPUT---------')
        print('Time Segment....{}\nFirst 5: {}\nLast 5: {}\n'.format(t.size, t[:5], t[-5:]))
        print('Frequency Segment....{}\nFirst 5: {}\nLast 5: {}\n'.format(f.size, f[:5], f[-5:]))
        print('Spectrogram Dim: {}\nF-Resolution: {}Hz/Band\nT-Resolution: {}'.format(Sxx.shape, f_res, t_res))

    if save or visualize:
        # plotting spectrogram
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        # display only 0Hz to 300kHz
        plt.ylim((0, vis_max_freq_range))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.title(save_title)
        plt.grid()
        plt.xlabel('Time [Sec]')
        plt.colorbar()

        if save:
            plt.savefig('result\{}.png'.format(save_title))

        if visualize:
            plt.show()

        plt.close()

    return t, f, Sxx


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


def one_dim_xcor_freq_band(mat1, mat2, verbose):
    '''
    We expect for both mat 1 n 2, shape[0] --> freq band, shape[1] --> time steps
    :param mat1: input
    :param mat2: input
    :param verbose: print the xcor map dimension
    :return: 2d normalized xcor map whr shape[0] --> freq band, shape[1] --> xcor steps
    '''
    # ensure they hv equal number of freq bands
    assert mat1.shape[0] == mat2.shape[0], 'Both matrix has different shape[0]'

    xcor_of_each_f_list = []
    # for all frequency bands
    for k in range(mat1.shape[0]):
        x_cor = np.correlate(mat1[k], mat2[k], 'full')
        xcor_of_each_f_list.append(x_cor)
    # xcor map of 2 phase map, axis[0] is freq, axis[1] is x-cor unit shift
    xcor_of_each_f_list = np.array(xcor_of_each_f_list)

    # normalize each xcor_map with linear function btw their max and min values
    scaler = MinMaxScaler(feature_range=(0, 1))
    xcor_of_each_f_list = scaler.fit_transform(xcor_of_each_f_list.ravel().reshape((-1, 1))) \
        .reshape((xcor_of_each_f_list.shape[0], xcor_of_each_f_list.shape[1]))

    if verbose:
        print('Xcor Map Dim (freq band, xcor steps): ', xcor_of_each_f_list.shape)

    return xcor_of_each_f_list


