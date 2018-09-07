import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, log, exp
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import gausspulse
# self defined library
from src.utils.dsp_tools import spectrogram_scipy, one_dim_xcor_2d_input
from src.utils.helpers import heatmap_visualizer, ProgressBarForLoop


def bumps(x):
    """
    A sum of bumps with locations t at the same places as jumps in blocks.
    The heights h and widths s vary and the individual bumps are of the
    form K(t) = 1/(1+|x|)**4
    """
    K = lambda x: (1. + np.abs(x)) ** -4.
    t = np.array([[.1, .13, .15, .23, .25, .4, .44, .65, .76, .78, .81]]).T
    h = np.array([[4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 2.1, 4.2]]).T
    w = np.array([[.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005]]).T

    return np.sum(h * K((x - t) / w), axis=0)


# --------------[Sine wave of increasing freq]--------------------
# https://stackoverflow.com/questions/19771328/
# sine-wave-that-exponentialy-changes-between-frequencies-f1-and-f2-at-given-time
def sweep_exponential(f_start, f_end, interval, n_steps):
    b = log(f_end/f_start) / interval
    a = 2 * pi * f_start / b
    for i in range(n_steps):
        delta = i / float(n_steps)
        t = interval * delta
        g_t = a * exp(b * t)
        print(t, 3 * sin(g_t))


def white_noise(time_axis, power):
    noise = np.random.normal(scale=np.sqrt(power), size=time_axis.size)

    return noise


def sine_wave_continuous(time_axis, amplitude, fo, phase=0):
    y = amplitude * np.sin(2 * np.pi * fo * time_axis + phase)

    return y


# equal to 5 seconds
def sine_pulse():
    fs = 10e3
    f_nyquist = fs / 2
    total_sample = 70e3
    time_axis = np.arange(total_sample) / fs
    zero = np.array([0]*30000)
    sine = 3 * np.sin(2*np.pi*1e3*time_axis[int(30e3):int(40e3)])

    pulse_sine = np.concatenate((zero, sine, zero), axis=0)

    spectrogram_scipy(pulse_sine,
                      fs=fs,
                      nperseg=500,
                      noverlap=100,
                      verbose=True,
                      return_plot=True,
                      vis_max_freq_range=fs/2)


def noise_time_shift_dataset(time_axis, fs, random_seed=None, num_series=2, normalize=True,
                             visualize_time_series=False, verbose=False):
    '''
    :param time_axis: White nosie will consists of (time_axis.size) points
    :param fs: sampling freq of the system
    :param num_series: num of diff random series,it controls the sample size
    :param normalize: Normalize with max min range for the Sxx matrix
    :param random_seed: If stated, the seed is fixed to that, orelse it is random everytime it is called
    :param visualize_time_series: Plot the time series for checking delay in time
    :param verbose: print the dimension of arranged FFT phase map
    :return: a 3d data set where shape[0]=sample sizes, shape[1]=sensor, shape[2]=freq
    AIM--------------
    Create a time series white noise and applied delay to them
    '''
    # Scaler declaration
    scaler = MinMaxScaler(feature_range=(0, 1))

    # whether u wat the random series to be same all the time
    if random_seed is None:
        pass
    else:
        np.random.seed(random_seed)

    random_set = []
    # generating different random series to increase sample size--------------------
    for i in range(num_series):
        # add as many shift as u want, fr small to big
        time_shift = [0, 100, 200, 300]  # 0.1, 0.2 .. seconds,
        noise = white_noise(time_axis=time_axis, power=1)

        signal = []
        for shift in time_shift:
            signal.append(np.concatenate((np.zeros(shift), noise), axis=0))
        # so that all time shifted series are of same length
        signal = pad_sequences(signal, maxlen=(signal[-1].size + 500), dtype='float32', padding='post')

        # visualize the time series signal after shift
        if visualize_time_series:
            # plot all raw signals
            diag_no = 1
            for s in signal:
                plt.subplot(6, 1, diag_no)
                plt.plot(s)
                diag_no += 1
            plt.show()
            plt.close()

        # sliced to take only 1-9 seconds
        signal_sliced = signal[:, 1000:9000]

        # convert all time series to F-T representation, form the phase map----------
        phase_map = []
        for s in signal_sliced:
            t, f, Sxx = spectrogram_scipy(s,
                                          fs=fs,
                                          nperseg=100,  # no of freq bin = nperseg/2 + 1
                                          noverlap=0,
                                          mode='angle',
                                          return_plot=False,
                                          verbose=False,
                                          vis_max_freq_range=fs/2)
            if normalize:
                # scaling of all phases from -pi to pi in 2d matrix of Sxx to 0-1
                Sxx = scaler.fit_transform(Sxx.ravel().reshape((-1, 1))).reshape((Sxx.shape[0], Sxx.shape[1]))
            phase_map.append(Sxx)
        # convert to ndarray
        phase_map = np.array(phase_map)
        # put all 3d phase map array into list
        random_set.append(phase_map)

    # concatenate all 3d ndarray in random_set list in axis 2
    phase_map = np.concatenate(random_set, axis=2)

    # data slicing and labelling------------------------------------------------------
    dataset, label = [], []
    class_no = 0

    # for all sensor pair (all classes)
    for i in range(phase_map.shape[0] - 1):
        # for all time steps (all samples)
        for j in range(phase_map.shape[2]):
            concat_phase = [phase_map[0, :, j], phase_map[i+1, :, j]]
            dataset.append(concat_phase)
            label.append(class_no)
        class_no += 1

    # convert to Ndarray
    dataset = np.array(dataset)
    label = np.array(label)

    if verbose:
        print('Original Data Dim (Sensor, Freq, Time): ', phase_map.shape)
        print('Paired Data set Dim (Sample size, Sensor, Freq): ', dataset.shape)
        print('Label Dim: ', label.shape)

    return dataset, label


def noise_time_shift_xcor_return(time_axis, fs, random_seed=None, num_series=1, normalize_xcor_score=True,
                                 visualize_time_series=False, visualize_xcor_map=False):
    '''
    :param time_axis: White nosie will consists of (time_axis.size) points
    :param fs: sampling freq of the system
    :param num_series: num of diff random series,it controls the sample size, class size = num_series
    :param normalize_xcor_score: Normalize with max min range for the xcor matrix
    :param random_seed: If stated, the seed is fixed to that, orelse it is random everytime it is called
    :param visualize_time_series: Plot the time series for checking delay in time
    :param visualize_xcor_map: Plot the color map for xcor map
    :return: a 3d data set where shape[0]=sample sizes, shape[1]=freq, shape[2]=xcor_steps
    AIM--------------
    Create a time series white noise -> time delay -> take phase component -> xcor -> xcor_map
    '''
    # ---------------------------[Declaration]-----------------------------
    # scaler declaration
    scaler = MinMaxScaler(feature_range=(0, 1))

    # whether u wat the random series to be same all the time
    if random_seed is None:
        pass
    else:
        np.random.seed(random_seed)

    class_1, class_2, class_3 = [], [], []
    # -------------------[Generating Noise with Shift]---------------------
    pb = ProgressBarForLoop(title='Generating Xcor Map Dataset', end=num_series)
    for i in range(num_series):
        # add as many shift as u want, fr small to big
        time_shift = [0, 100, 200, 300]  # 0.1, 0.2 .. seconds,
        noise = white_noise(time_axis=time_axis, power=1)

        signal = []
        for shift in time_shift:
            signal.append(np.concatenate((np.zeros(shift), noise), axis=0))
        # so that all time shifted series are of same length
        signal = pad_sequences(signal, maxlen=(signal[-1].size + 500), dtype='float32', padding='post')

        # visualize the shifted time series signal -------------------------
        if visualize_time_series:
            # plot all raw signals
            diag_no = 1
            for s in signal:
                plt.subplot(6, 1, diag_no)
                plt.plot(s)
                diag_no += 1
            plt.show()
            plt.close()

        # sliced to take only 1-9 seconds
        signal_sliced = signal[:, 1000:9000]

        # -------------------[Converting time signal to F-T representation]---------------------
        phase_map = []
        for s in signal_sliced:
            t, f, Sxx = spectrogram_scipy(s,
                                          fs=fs,
                                          nperseg=100,  # no of freq bin = nperseg/2 + 1
                                          noverlap=0,  # no overlapped signal in each window take
                                          mode='angle',
                                          return_plot=False,
                                          verbose=False,
                                          vis_max_freq_range=fs / 2)
            phase_map.append(Sxx)
        # convert to ndarray
        phase_map = np.array(phase_map)

        # -------------------[Converting 2 phase map to 1 X-cor map]---------------------
        xcor_of_each_f_list = []
        # for map 1, 2, 3 to correlate with map 0
        for j in range(1, phase_map.shape[0], 1):
            # for all frequency bands
            for k in range(phase_map.shape[1]):
                x_cor = np.correlate(phase_map[0, k], phase_map[j, k], 'full')
                xcor_of_each_f_list.append(x_cor)
            # xcor map of 2 phase map, axis[0] is freq, axis[1] is x-cor unit shift
            xcor_of_each_f_list = np.array(xcor_of_each_f_list)
            if normalize_xcor_score:
                # normalize each xcor_map with linear function btw their max and min values
                xcor_of_each_f_list = scaler.fit_transform(xcor_of_each_f_list.ravel().reshape((-1, 1)))\
                    .reshape((xcor_of_each_f_list.shape[0], xcor_of_each_f_list.shape[1]))

            # Print all xcor_map
            if visualize_xcor_map:
                heatmap_visualizer(x_axis=np.arange(1, xcor_of_each_f_list.shape[1] + 1, 1),
                                   y_axis=f,
                                   zxx=xcor_of_each_f_list,
                                   label=['Xcor_steps', 'Frequency', 'Correlation Score'],
                                   output='color_map')
            # put into different classes according to their shift
            if j is 1:
                class_1.append(xcor_of_each_f_list)
            if j is 2:
                class_2.append(xcor_of_each_f_list)
            if j is 3:
                class_3.append(xcor_of_each_f_list)

            # empty n reset the xcor_of_each_f_list
            xcor_of_each_f_list = []
        pb.update(now=i)
    pb.destroy()
    # -------------------[Data Slicing and Train Test data]---------------------
    class_1 = np.array(class_1)
    class_2 = np.array(class_2)
    class_3 = np.array(class_3)
    all_class = [class_1, class_2, class_3]
    dataset = np.concatenate(all_class, axis=0)
    label = np.array([0]*class_1.shape[0] + [1]*class_2.shape[0] + [2]*class_3.shape[0])

    print('Data set Dim: ', dataset.shape)
    print('Label Dim: ', label.shape)

    return dataset, label


def gauss_pulse_timeshift_dataset(class_sample_size, visualize_each_data_in_time=False):
    '''
    Designing 3 Gauss pulse with different time shift, contaminated with white noise
    This return 2 classes of xcor map, created from xcor signal[0] & [1] and signal[0] & [2]. 2 classes repr.
    different shift of the pulse in the received signal in sensors.
    :return:
    a 3d array, where axis[0] -> samples size
                      axis[1] -> freq axis
                      axis[2] -> xcor scores
    # This data set was successfully trained by CNN, proving that this pattern is recognizable
    '''
    # time setting
    t = np.linspace(0, 2, 5000, endpoint=False)

    mix_signal_1, mix_signal_2, mix_signal_3 = [], [], []
    # creating n samples for each of the 2 classes
    for i in range(class_sample_size):
        # create noise contaminated gauss pulse
        mix_signal_1.append(gausspulse(t - 0.5, fc=50) + white_noise(time_axis=t, power=0.1))
        mix_signal_2.append(gausspulse(t - 0.6, fc=50) + white_noise(time_axis=t, power=0.1))
        mix_signal_3.append(gausspulse(t - 0.7, fc=50) + white_noise(time_axis=t, power=0.1))

    class_1, class_2 = [], []
    for i in range(class_sample_size):
        # STFT
        t, f, mat1, _ = spectrogram_scipy(sampled_data=mix_signal_1[i],
                                          fs=2500,  # because 2500 points per sec
                                          nperseg=200,
                                          noverlap=0,
                                          mode='magnitude',
                                          return_plot=False,
                                          verbose=False)

        _, _, mat2, _ = spectrogram_scipy(sampled_data=mix_signal_2[i],
                                          fs=2500,  # because 2500 points per sec
                                          nperseg=200,
                                          noverlap=0,
                                          mode='magnitude',
                                          return_plot=False,
                                          verbose=False)

        _, _, mat3, _ = spectrogram_scipy(sampled_data=mix_signal_3[i],
                                          fs=2500,  # because 2500 points per sec
                                          nperseg=200,
                                          noverlap=0,
                                          mode='magnitude',
                                          return_plot=False,
                                          verbose=False)
        l = np.array([mat1, mat2, mat3])
        xcor_map = one_dim_xcor_2d_input(input_mat=l, pair_list=[(0, 1), (0, 2)], verbose=False)
        # signal labelling
        class_1.append(xcor_map[0])
        class_2.append(xcor_map[1])
        # fig1 = three_dim_visualizer(x_axis=np.arange(0, xcor_map.shape[2], 1),
        #                             y_axis=f,
        #                             zxx=xcor_map[0],
        #                             output='2d',
        #                             label=['a', 'b', 'cx'])
        # fig2 = three_dim_visualizer(x_axis=np.arange(0, xcor_map.shape[2], 1),
        #                             y_axis=f,
        #                             zxx=xcor_map[1],
        #                             output='2d',
        #                             label=['a', 'b', 'cx'])
        # plt.show()

    # convert to np array
    class_1 = np.array(class_1)
    class_2 = np.array(class_2)
    dataset = np.concatenate((class_1, class_2), axis=0)
    label = np.array([0] * class_1.shape[0] + [1] * class_2.shape[0])

    print('Data set Dim: ', dataset.shape)
    print('Label Dim: ', label.shape)

    return dataset, label
