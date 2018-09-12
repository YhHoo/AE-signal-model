from progressbar import ProgressBar, Percentage, Bar, SimpleProgress, ETA
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import model_from_json
import keras.backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # do not delete
import numpy as np
from nptdms import TdmsFile
from os import listdir
from matplotlib import cm
import sys
from vispy import app, visuals, scene
# self lib
from src.utils.dsp_tools import fft_scipy


class ProgressBarForLoop:
    '''
    Try out more progressbar from https://github.com/coagulant/progressbar-python3/blob/master/examples.py
    progress bar, maxval is like max value in a ruler, and set the progress with update()
    The 'now' starts from 0
    '''
    # progress bar setup, set the title and max value
    def __init__(self, title, end=100):
        print(title + ':')
        widgets = [Percentage(), ' ',
                   Bar(marker='#', left='[', right=']'),
                   ' ', SimpleProgress(), ' --> ', ETA()]
        self.pbar = ProgressBar(widgets=widgets, maxval=end).start()

    def update(self, now):
        self.pbar.update(now+1)

    # kill the bar, ready to start over the new one
    def destroy(self):
        self.pbar.finish()


class ModelLogger:
    '''
    model_name practice --> [test_no]_[architecture]_[date]
    e.g. test2_CNN_22_5_18

    # FOR SAVING MODEL-----------------------------
    save_architecture() -> saves the architecture only into .JSON
    save_best_weight_checkpoint() -> return a checkpoint to be placed into model.fit() so
    it saves the best weights during training

    # FOR PLOTTING AND SAVING LEARNING CURVE-------
    learning_curve() -> take in history object ruturned during fit() and plot and save the fig
    '''

    def __init__(self, model, model_name):
        self.model = model
        self.path = direct_to_dir(where='saved_model') + model_name

    def save_architecture(self, save_readable=False):
        # serialize and saving the model structure to JSON
        model_json = self.model.to_json()
        model_json_pretty = self.model.to_json(indent=4)
        # path to save
        path = self.path + '.json'
        # saving
        with open(path, 'w') as json_file:
            json_file.write(model_json)
        print('Architecture saved -->{}.json'.format(path))
        # save the model.summary() into txt
        if save_readable:
            path = self.path + '.txt'
            with open(path, 'w') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
                f.write(model_json_pretty)
        print('Readable Architecture saved -->{}.txt'.format(path))

    def save_best_weight_cheakpoint(self, monitor='val_loss', period=1):
        '''
        :param monitor: value to monitor when saving
        :param mode: what value should be save
        :param period: how frequent to check and save
        :return: a callback_list to be placed on fit(callbacks=...)
        '''
        path = self.path + '.h5'
        # automate mode
        if monitor is 'val_loss':
            mode = 'min'
        elif monitor is 'val_acc':
            mode = 'max'

        checkpoint = ModelCheckpoint(filepath=path,
                                     monitor=monitor,
                                     verbose=1,
                                     save_best_only=True,
                                     mode=mode,  # for acc, it should b 'max'; for loss, 'min'
                                     period=period)  # no of epoch btw checkpoints
        return checkpoint

    # this function use the model history returned by fit() to plot learning curve and save it
    def learning_curve(self, history, save=False, show=False, title='Learning Curve'):
        '''

        :param history: hist object returned by fit()
        :param save: save figure
        :param show: show the plot, set to False for overnight run
        :param title: fig title
        :return: :return: None, but it will save fig of learning curve
        '''
        fig = plt.figure(figsize=(6, 4))
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.plot(history.history['acc'], label='train_acc')
        plt.plot(history.history['val_acc'], label='val_acc')
        plt.legend()
        plt.grid()
        plt.title(title)
        if save:
            fig.savefig(self.path + '_LrCurve.png')
        if show:
            plt.show()
        # free up memory
        plt.close('all')

    def save_recall_precision_f1(self, y_true, y_pred, all_class_label):
        '''
        :param y_true: list or 1d array of actual labels
        :param y_pred: list or 1d array of model prediction (after argmax())
        :param all_class_label: list or 1d array of integers for labelling the class, e.g. 0, 1, 2, 3, ...
        :return: None, but it will save 2 csv of confusion matrix and recall_precision table
        '''
        # Consider this as a wrapper for the recall_precision_multiclass(), for easy saving purpose
        mat, r, p, f1 = compute_recall_precision_multiclass(y_true=y_true, y_pred=y_pred, all_class_label=all_class_label)

        mat_filename = self.path + '_confusion_mat.csv'
        recall_precision_df_filename = self.path + '_recall_prec_f1.csv'

        # prepare and save confusion matrix
        mat.to_csv(mat_filename)

        # prepare and save each class recall n precision n f1
        mat = np.array([r, p])
        recall_precision_df = pd.DataFrame(mat,
                                           index=['recall', 'precision'],
                                           columns=all_class_label)
        recall_precision_df.loc['f1'] = None
        recall_precision_df.iloc[2, 0] = f1
        recall_precision_df.to_csv(recall_precision_df_filename)


def load_model(model_name=None):
    '''
    :param model_name: The model name
    :param dir: The location that contains .h5, .json of the model
    :return: a model loaded with .h5 and .json
    AIM: this just simplifies the model loading procedure by wrapping them in one.
    This has to be followed by model.compile() if we wish to train the model later
    '''
    path = direct_to_dir(where='saved_model') + model_name

    # load architecture from .json
    with open(path + '.json', 'r') as f:
        model = model_from_json(f.read())

    # load weights from .h5
    model.load_weights(path + '.h5')
    print('Model Loaded !')

    return model


def evaluate_model_for_all_class(model, test_x, test_y):
    '''
    :param model: trained model
    :param test_x: 2d ndarray
    :param test_y: in one-hot encoding
    :return:
    '''
    # manual prediction, convert the output from one-hot encoding bac to class no
    # e.g. [0 1 0 0] --> 1, [0 0 1 0] --> 2
    # Make sure classes in the test_y are at subsequent order
    prediction = model.predict(test_x)
    prediction = np.argmax(prediction, axis=1)
    actual = np.argmax(test_y, axis=1)

    # visualize the multiclass classification accuracy
    plt.plot(actual, color='r', label='Actual')
    plt.plot(prediction, color='b', label='Prediction', linestyle='None', marker='x')
    plt.title('Classifier Evaluation in Visual')
    plt.legend()
    plt.show()


def reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y, fourth_dim=1, num_classes=None, verbose=False):
    '''
    :param train_x: Expecting a 3d np array, where shape[0] is sample size
    :param train_y: Expecting 1d np array
    :param test_x: Expecting a 3d np array, where shape[0] is sample size
    :param test_y: Expecting 1d np array
    :param fourth_dim: Usually is 1, for non RGB image data
    :param num_classes: For converting 1, 2, 3 into binary [1,0,0], [0,1,0], [0,0,1]
    :param verbose: Print the returned dimension
    :return: all reshaped train and test data, ready to fit into Conv2d
    '''
    train_x_4d = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], fourth_dim))
    test_x_4d = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], fourth_dim))

    # to categorical
    train_y_cat = to_categorical(train_y, num_classes=num_classes)
    test_y_cat = to_categorical(test_y, num_classes=num_classes)

    if verbose:
        print('-----RESHAPED------')
        print('Train_x Dim: ', train_x_4d.shape)
        print('Test_x Dim: ', test_x_4d.shape)
        print('Train_y Dim:', train_y_cat.shape)
        print('Test_y Dim:', test_y_cat.shape)

    return train_x_4d, train_y_cat, test_x_4d, test_y_cat


def break_balanced_class_into_train_test(input, label, num_classes, shuffled_each_class=True,
                                         train_split=0.7, verbose=False):
    '''
    :param input: expect a 3d np array where 1st index is total sample size
    :param label: expect a 1d np array of same size as input.shape[0]
    :param num_classes: total classes to break into
    :param shuffled_each_class: it will shuffle the samples in every class
    :param verbose: print the summary of train test size
    :return: a train and test set in ndarray

    AIM----------------------------------
    This is when we receive a list of N classes samples(a list of 2D array) all concatenate together sequentially
    e.g [0,..,0,1,..1,2,..,2...N-1..N-1] and we want to split them into train and test.

    WARNING------------------------------
    Every class size have to be EQUAL !

    EXAMPLE------------------------------(execute it and watch)
    data = np.array([[[1, 2],
                      [3, 4]],
                     [[2, 3],
                      [4, 5]],
                     [[3, 4],
                      [5, 6]],
                     [[11, 12],
                      [13, 14]],
                     [[12, 13],
                      [14, 15]],
                     [[13, 14],
                      [15, 16]]])
    label = np.array([0, 0, 0, 1, 1, 1])
    train_x, train_y, test_x, test_y = break_into_train_test(input=data, label=label,
                                                             num_classes=2, train_split=0.7, verbose=True)
    print('Train x:\n', train_x)
    print('Train y:\n', train_y)
    print('Test x:\n', test_x)
    print('Test y:\n', test_y)
    '''
    # ensure both input and label sample size are equal
    assert input.shape[0] == label.shape[0], 'Sample size of Input and Label must be equal !'

    # shuffling work
    if shuffled_each_class:
        class_split_index = np.linspace(0, input.shape[0], num_classes + 1)

        # accessing index btw each classes
        for i in range(class_split_index.size - 1):
            # for class of index 0-10, this array will return [0, 1, ...9]
            entire_class_index = np.arange(class_split_index[i], class_split_index[i + 1], 1)
            # convert to int from float
            entire_class_index = [int(i) for i in entire_class_index]
            # shuffle the index [0, 1, ...9] --> [4, 3, ...7]
            entire_class_index_shuffled = np.random.permutation(entire_class_index)
            # shuffle the value of the class and store the shuffled values
            class_data_shuffled = input[entire_class_index_shuffled]
            # replace the original unshuffled matrix
            input[entire_class_index] = class_data_shuffled

    # slicing work
    sample_size = input.shape[0]
    # create an index where the
    class_break_index = np.linspace(0, sample_size, num_classes + 1)
    # convert from float to int
    class_break_index = [int(i) for i in class_break_index]
    # determine split index from first 2 items of class_break_index list
    split_index_from_start = int(train_split * (class_break_index[1] - class_break_index[0]))

    # training set
    train_x, test_x, train_y, test_y = [], [], [], []
    # slicing in btw every intervals for classes
    for i in range(len(class_break_index) - 1):
        train_x.append(input[class_break_index[i]: (class_break_index[i] + split_index_from_start)])
        test_x.append(input[(class_break_index[i] + split_index_from_start): class_break_index[i + 1]])
        train_y.append(label[class_break_index[i]: (class_break_index[i] + split_index_from_start)])
        test_y.append(label[(class_break_index[i] + split_index_from_start): class_break_index[i + 1]])

    # convert list of list into just a list
    train_x = [data for classes in train_x for data in classes]
    test_x = [data for classes in test_x for data in classes]
    train_y = [data for classes in train_y for data in classes]
    test_y = [data for classes in test_y for data in classes]

    # convert list to np array
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    if verbose:
        print('\n----------TRAIN AND TEST SET---------')
        if shuffled_each_class:
            print('-------------[Shuffled]--------------')
        print('Train_x Dim: ', train_x.shape)
        print('Test_x Dim: ', test_x.shape)
        print('Train_y Dim:', train_y.shape)
        print('Test_y Dim:', test_y.shape)

    # return
    return train_x, train_y, test_x, test_y


def heatmap_visualizer(x_axis, y_axis, zxx, label=('x', 'y', 'z'), output='2d',
                       title='None', vis_range=[None, None, None, None]):
    '''
    Recommend for heatmap
    :param x_axis: the actual x-axis we wish to see in cartesian plane
    :param y_axis: the actual y-axis we wish to see in cartesian plane
    :param zxx: Zxx is a matrix of dim: (y_axis.size, x_axis.size)
    :param label: List of string labels for [x_axis, y_axis, z_axis]
    :param output: bar_chart or color_map output
    :param vis_range: axis[0] and [1] is y-axis min n max, axis[2] and [3] is x-axis min n max
    Note that they only applies for visualization, the data is fully generated anyway.
    :param title: title on top of the plot
    :return: plot()
    '''
    # make sure the axes are of equal sizes for zxx
    assert x_axis.size == zxx.shape[1], 'axis [1] of zxx differ from x_axis.size'
    assert y_axis.size == zxx.shape[0], 'axis [0] of zxx differ from y_axis.size'
    assert output is not None, 'Please specify Output'
    # print('Visualizing...')

    if output is '3d':
        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111, projection='3d')
        for i in range(y_axis.size):
            ax.bar(x_axis, zxx[i], zs=y_axis[i], zdir='y', alpha=0.8)
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_zlabel(label[2])
    elif output is '2d':
        fig = plt.figure(figsize=(9, 5))
        fig.suptitle(title)
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
        colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
        i = ax.pcolormesh(x_axis, y_axis, zxx)
        fig.colorbar(i, cax=colorbar_ax)
        ax.grid()
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_ylim(bottom=vis_range[0], top=vis_range[1], auto=True)
        ax.set_xlim(left=vis_range[2], right=vis_range[3], auto=True)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # plt.figure(figsize=(9, 5))  # 8 inches square
        # plt.title(title)
        # plt.pcolormesh(x_axis, y_axis, zxx)
        # plt.xlabel(label[0])
        # plt.set_ylim()
        # plt.ylabel(label[1])
        # plt.colorbar()

    # set to false when we wan to save a series of plot
    return fig


def read_all_tdms_from_folder(folder_path=None):
    '''
    :param folder_path: The folder which contains several sets data of same setup (Test rig)
    :return: 3d matrix where shape[0]=no. of sets | shape[1]=no. of AE Signal points | shape[2]=no. of sensors
    Aim: To combine all sets of data for same experiment setup into one 3d array.
    WARNING: All sets of input data must contains same number of points e.g. 5 seconds/5M points for all sets
    '''
    # ensure path exist
    assert folder_path is not None, 'No Folder is selected'

    # list full path of all tdms file in the specified folder
    all_file_path = [(folder_path + f) for f in listdir(folder_path) if f.endswith('.tdms')]
    n_channel_matrix = []

    # do for all 3 sets of tdms file
    # read tdms and save as 4 channel np array
    pb = ProgressBarForLoop('\nReading --> ' + folder_path, end=len(all_file_path))
    for f in all_file_path:
        tdms_file = TdmsFile(f)
        tdms_df = tdms_file.as_dataframe()
        # store the df values to list
        n_channel_matrix.append(tdms_df.values)
        # update progress
        pb.update(now=all_file_path.index(f))
    # kill progress bar
    pb.destroy()
    # convert the list matrix
    n_channel_matrix = np.array(n_channel_matrix)
    print('Acquired Data Dim: ', n_channel_matrix.shape, '\n')

    return n_channel_matrix


def read_single_tdms(filename=None):
    '''
    While read_all_tdms_from_folder() read all tdms file from a folder, when a folder contains huge no of tdms, it is
    time consuming to read all when we only want to visualize/test with one single tdms file
    :param filename: file name and path of the specific tdms file
    :return:
    '''
    print('Reading --> ', filename)
    tdms_file = TdmsFile(filename)
    tdms_df = tdms_file.as_dataframe()
    # store the df values to list
    n_channel_matrix = tdms_df.values
    # convert the list matrix
    n_channel_matrix = np.array(n_channel_matrix)
    print('Acquired Data Dim: ', n_channel_matrix.shape, '\n')

    return n_channel_matrix


def plot_multiple_timeseries(input, subplot_titles, main_title):
    '''
    Aspect axis[0] of input is no. of sensors/diff features, axis[1] is time steps. All time series has to be
    SAME length !
    :param input: a 2d-array / list
    :param subplot_titles: title for every plot
    :param main_title: the big title
    :return: rectangular fig obj
    '''
    no_of_plot = len(input)
    fig = plt.figure(figsize=(5, 8))
    fig.suptitle(main_title, fontweight="bold", size=8)
    fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.03)
    # first plot
    ax1 = fig.add_subplot(no_of_plot, 1, 1)
    ax1.plot(input[0])
    ax1.set_title(subplot_titles[0], size=8)
    ax1.set_ylim(bottom=-1, top=1)
    # the rest of the plot
    for i in range(1, no_of_plot, 1):
        ax = fig.add_subplot(no_of_plot, 1, i+1, sharex=ax1)  # add in sharey=ax1 if wan to share y axis too
        ax.plot(input[i])
        ax.set_title(subplot_titles[i], size=8)
        ax.set_ylim(bottom=-1, top=1)
    return fig


def plot_multiple_timeseries_with_roi(input, subplot_titles, main_title, peak_center_list, roi_width=None):
    '''
    Aspect axis[0] of input is no. of sensors/diff features, axis[1] is time steps. All time series has to be
    SAME length !
    :param input: a 2d-array / list
    :param subplot_titles: title for every plot
    :param main_title: the big title
    :param peak_center_list: a list of 1d array, where len(peak_list) is no of channels, and 1d-array inside is index
                      of the peak in particular channel
    :param roi_width: tuple of 2, e.g. (a, b) --> the roi will b [c-a, c+b]
    :return: rectangular fig obj
    '''
    no_of_plot = len(input)
    fig = plt.figure(figsize=(5, 8))
    fig.suptitle(main_title, fontweight="bold", size=8)
    fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.03)
    # first plot
    ax1 = fig.add_subplot(no_of_plot, 1, 1)
    ax1.plot(input[0])
    ax1.plot(peak_center_list[0], input[0][peak_center_list[0]], marker='o', ls='', ms=3, mfc='red')

    # roi region highlighting
    for peak in peak_center_list[0]:
        ax1.axvspan(xmin=peak - roi_width[0],
                    xmax=peak + roi_width[1],
                    color='red', alpha=0.5)

    ax1.set_title(subplot_titles[0], size=8)
    ax1.set_ylim(bottom=-1, top=1)

    # the rest of the plot
    for i in range(1, no_of_plot, 1):
        ax = fig.add_subplot(no_of_plot, 1, i+1, sharex=ax1)  # add in sharey=ax1 if wan to share y axis too
        ax.plot(input[i])
        ax.plot(peak_center_list[i], input[i][peak_center_list[i]], marker='o', ls='', ms=3, mfc='red')

        # roi region highlighting
        for peak in peak_center_list[i]:
            ax.axvspan(xmin=peak - roi_width[0],
                       xmax=peak + roi_width[1],
                       color='red', alpha=0.5)

        ax.set_title(subplot_titles[i], size=8)
        ax.set_ylim(bottom=-1, top=1)
    return fig


def plot_multiple_level_decomposition(ori_signal, dec_signal, dec_level, main_title, fs):
    '''
    Recommendation:
    This fn plot the output from pywt.wavdec(), ideal for plotting original and dec signal on the left column, tgt
    with their fft on the right column.

    Generally:
    It will append the ori_signal and dec_signal into a list. Then find fft of each of them, plot all of them on 2 col

    Warning:
    Time series signal need not be same length, but the axis values will be different. all signal will
    b stretched to the same length.

    :param ori_signal: 1d-array
    :param dec_signal: a 2d-array / list of 1d array (** i.e. output of pywt.wavedec() )
    :param dec_titles: title for every dec, starting fr
    :param main_title: the big title
    :return: fig obj
    '''

    fig = plt.figure(figsize=(5, 8))
    fig.suptitle(main_title, fontweight="bold")
    fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.03, left=0.08, right=0.95)

    # append all signal into a list
    all_signal = list()
    all_signal.append(ori_signal)
    for arr in dec_signal:
        all_signal.append(arr.tolist())

    no_of_column_plot = len(all_signal)

    # create labels for plots
    plot_label = ['original'] + ['cA_{}'.format(dec_level)] + ['cD_{}'.format(i) for i in range(dec_level, 0, -1)]

    # fft for all signal
    fft_mag_store = list()
    f_axis_store = list()
    for signal in all_signal:
        f_mag, _, f_axis = fft_scipy(sampled_data=signal, fs=fs, visualize=False)
        fft_mag_store.append(f_mag)
        f_axis_store.append(f_axis)

    # plot all
    for i in range(0, no_of_column_plot, 1):
        # left column, 1, 3, 5, 7 ... for dec signal
        ax = fig.add_subplot(no_of_column_plot, 2, i*2 + 1)
        ax.plot(all_signal[i])
        ax.grid(linestyle='dotted')
        ax.set_title(plot_label[i], size=8)

        # odd plot 1, 3, 5, ... for fft of dec signal
        ax = fig.add_subplot(no_of_column_plot, 2, i*2 + 2)
        ax.plot(f_axis_store[i], fft_mag_store[i])
        ax.grid(linestyle='dotted')
        ax.set_title(plot_label[i], size=8)
        ax.set_xlabel('Freq (Hz)')
        ax.set_ylabel('Amplitude')

        # use scitific notation on y axis of fft plot
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    return fig


def plot_simple_heatmap(zxx):
    '''
    :param zxx: a 2d array input
    :return: a heatmap plot of the zxx
    '''
    print('Input Dim Detected: ', zxx.shape)
    fig = plt.imshow(zxx, interpolation='None', cmap='jet')

    return fig


def plot_two_heatmap_in_one_column(input_1, input_2):
    '''
    This function plot 2 heatmap (e.g. xcor map, FT map) in one fig (up and down, in one column), for comparison.
    For best comparison experience by visual, the 2 inputs shud be equal dimension.
    Note that the values in axis[0] will be plotted top down (for small to big)
    :param input_1: 2d array
    :param input_2: 2d array
    :return: fig up and down plot
    Suggestion: let Shape[0] always be freq, shape[1] be xcor steps/ time steps
    '''
    val_test = [input_1, input_2]

    fig = plt.figure(figsize=(5, 7))
    fig.suptitle('Xcor Map of 2 leak class')
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(2, 1),
                    axes_pad=0.05,
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single")

    for val, ax in zip(val_test, grid):
        im = ax.imshow(val, vmin=0, vmax=1, extent=(0.1, 0.41, 0.6, 0.39))  # (left, right, bottom, top)

    grid.cbar_axes[0].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(False)

    return fig


def plot_heatmap_series_in_one_column(zxx_list, title='No Title', subplot_title='No Title'):
    '''
    **Preferably rectangle 2d heatmap, e.g. (10, 300) or (50, 300)
    This plot a series of rectangular heatmap in a drop down (1 column) format, with a shared color bar.
    This is a extended version of dual_heatmap_plot()
    :param zxx_list: list of 2d array input
    :param title: Big title
    :param subplot_title: title for each of the subplot
    :return: fig object of the multiple plot in one
    '''
    # assert type(zxx_list) == list, 'zxx_list MUST be list dtype'
    no_of_component = len(zxx_list)
    fig = plt.figure(figsize=(5, 7))
    fig.suptitle(title)
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(no_of_component, 1),
                    axes_pad=0.1,
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size='1%')

    subplot_title_index = 0
    for val, ax in zip(zxx_list, grid):
        title = '{}_[{}]'.format(subplot_title, subplot_title_index)
        # this configure titles for each heat map
        ax.set_title(title, position=(-0.15, 0.9), fontsize=7, rotation='vertical')
        # this configure the dimension of the heat map in the fig object
        im = ax.imshow(val, vmin=0, vmax=1, extent=(0.01, 0.91, 0.6, 0.39), cmap='jet')  # (left, right, bottom, top)
        subplot_title_index += 1

    # this simply add color bar instance
    grid.cbar_axes[0].colorbar(im)

    # this toggle labels for color bar
    for cax in grid.cbar_axes:
        cax.toggle_label(True)

    return fig


def plot_heatmap_series_in_four_column(column_1_heatmap, column_2_heatmap, column_3_heatmap, column_4_heatmap,
                                       main_title='No Title', each_column_title=['None', 'None', 'None', 'None'],
                                       each_subplot_title=['None', 'None', 'None']):
    '''
    **For visualizing the layer activation of CNN
    this is for plotting the input 2d array as image(heatmap) in first column, followed by the activation result by
    different filters in following 3 layers of CNN

    :param column_1_heatmap: only a single 2d ndarray input
    :param column_2_heatmap: a list of 2d ndarray
    :param column_3_heatmap: a list of 2d ndarray
    :param column_4_heatmap: a list of 2d ndarray
    :param main_title: title of the figure
    :param each_column_title: a list of string for every column title (list of 4 only, for col 1-4)
    :param each_subplot_title: a list of string (only a same name for every column, an index will be appended after
                               the name for subplot in every column) (list of 3 only, for col 2-4)
    :return: a fig of all plots
    '''
    # getting the no of heatmap
    no_of_heatm_in_column_2 = len(column_2_heatmap)
    no_of_heatm_in_column_3 = len(column_3_heatmap)
    no_of_heatm_in_column_4 = len(column_4_heatmap)

    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0.06, right=0.96)
    # main title of figure
    fig.suptitle(main_title)
    # all axes grid's big title
    fig.text(0.10, 0.9, each_column_title[0])
    fig.text(0.35, 0.9, each_column_title[1])
    fig.text(0.58, 0.9, each_column_title[2])
    fig.text(0.82, 0.9, each_column_title[3])

    grid_0 = AxesGrid(fig, 141,
                      nrows_ncols=(1, 1),
                      axes_pad=0.1,
                      share_all=True,
                      label_mode="L",
                      cbar_location="bottom",
                      cbar_mode="single",
                      cbar_size='15%')

    grid_1 = AxesGrid(fig, 142,
                      nrows_ncols=(no_of_heatm_in_column_2, 1),
                      axes_pad=0.1,
                      share_all=True,
                      label_mode="L",
                      cbar_location="right",
                      cbar_mode="single",
                      cbar_size='0.5%')
    grid_2 = AxesGrid(fig, 143,
                      nrows_ncols=(no_of_heatm_in_column_3, 1),
                      axes_pad=0.1,
                      share_all=True,
                      label_mode="L",
                      cbar_location="right",
                      cbar_mode="single",
                      cbar_size='0.5%')
    grid_3 = AxesGrid(fig, 144,
                      nrows_ncols=(no_of_heatm_in_column_4, 1),
                      axes_pad=0.1,
                      share_all=True,
                      label_mode="L",
                      cbar_location="right",
                      cbar_mode="single",
                      cbar_size='0.5%')

    for ax in grid_0:
        im = ax.imshow(column_1_heatmap, vmin=0, vmax=1, extent=(0.01, 0.91, 0.6, 0.39), cmap='jet')

    for val, ax in zip(column_2_heatmap, grid_1):
        # this configure titles for each heat map
        ax.set_title(each_subplot_title[0], position=(-0.15, 0.388), fontsize=7, rotation='vertical')
        # this configure the dimension of the heat map in the fig object
        im = ax.imshow(val, vmin=0, vmax=1, extent=(0.01, 0.91, 0.6, 0.39), cmap='jet')  # (left, right, bottom, top)

    for val, ax in zip(column_3_heatmap, grid_2):
        # this configure titles for each heat map
        ax.set_title(each_subplot_title[1], position=(-0.15, 0.388), fontsize=7, rotation='vertical')
        # this configure the dimension of the heat map in the fig object
        im = ax.imshow(val, vmin=0, vmax=1, extent=(0.01, 0.91, 0.6, 0.39), cmap='jet')  # (left, right, bottom, top)

    for val, ax in zip(column_4_heatmap, grid_3):
        # this configure titles for each heat map
        ax.set_title(each_subplot_title[2], position=(-0.15, 0.388), fontsize=7, rotation='vertical')
        # this configure the dimension of the heat map in the fig object
        im = ax.imshow(val, vmin=0, vmax=1, extent=(0.01, 0.91, 0.6, 0.39), cmap='jet')  # (left, right, bottom, top)

    # this simply add color bar instance
    grid_0.cbar_axes[0].colorbar(im)
    grid_1.cbar_axes[0].colorbar(im)
    grid_2.cbar_axes[0].colorbar(im)
    grid_3.cbar_axes[0].colorbar(im)

    # this toggle labels for color bar
    for cax in grid_0.cbar_axes:
        cax.toggle_label(True)
    for cax in grid_1.cbar_axes:
        cax.toggle_label(True)
    for cax in grid_2.cbar_axes:
        cax.toggle_label(True)
    for cax in grid_3.cbar_axes:
        cax.toggle_label(True)

    return fig


def compute_recall_precision_multiclass(y_true, y_pred, all_class_label, verbose=True):
    '''
    Reference website: http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html
    :param y_true: list or 1d array of actual labels
    :param y_pred: list or 1d array of model prediction (after argmax())
    :param all_class_label: list or 1d array of integers for labelling the class, e.g. 0, 1, 2, 3, ...
    :param verbose: Print out the recall and precision for each class
    :return: conf_mat --> a square dataframe of pandas
             recall_each_class --> list or 1d array of integers (follow order of all_class_label)
             precision_each_class --> list or 1d array of integers (follow order of all_class_label)
    '''
    # create labels for index and columns of confusion matrix
    col_labels = ['Actual_Class[{}]'.format(i) for i in all_class_label]
    index_labels = ['Predict_Class[{}]'.format(i) for i in all_class_label]

    # arrange all prediction and actual label into confusion matrix
    data = confusion_matrix(y_true=y_true, y_pred=y_pred)
    conf_mat = pd.DataFrame(data=data.T, index=index_labels, columns=col_labels)

    # taking all diagonals values into a 1d array
    diag = np.diag(conf_mat.values)

    # sum across rows and columns of confusion mat
    total_pred_of_each_class = pd.DataFrame.sum(conf_mat, axis=1).values
    total_samples_of_each_class = pd.DataFrame.sum(conf_mat, axis=0).values

    # Recall = TP_A/(TP_A+FN_A) ; Precision = TP_A/(TP_A+FP_A)
    recall_each_class = diag / total_samples_of_each_class
    precision_each_class = diag / total_pred_of_each_class

    # f1 score (own method, pls verify wif xinjie)
    recall_avg = np.average(recall_each_class)
    precision_avg = np.average(precision_each_class)
    # average
    f1_score = (2*recall_avg*precision_avg) / (recall_avg + precision_avg)
    if verbose:
        print('class recall: ', recall_each_class)
        print('class precision: ', precision_each_class)
        print('Overall F1-score: ', f1_score)

    return conf_mat, recall_each_class, precision_each_class, f1_score


def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    '''
    retrieved from https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
    :param model: Keras model
    :param model_inputs: Model inputs for which we want to get the activations (e.g. 200 MNIST images)
    :param print_shape_only: if TRUE print shape of activations arrays of every layer only. If false it will print
                             entire array
    :param layer_name: layer where we want to read the activation. If none, it will return all layer activation
    :return: a list of np array where len(return) = no of layers of interest (by layer name)
    '''
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs (in tensor)

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)

    return activations


def direct_to_dir(where=None):
    if where is 'result':
        return 'C:/Users/YH/PycharmProjects/AE-signal-model/result/'
    elif where is 'tf_graph':
        return 'C:/Users/YH/PycharmProjects/AE-signal-model/result/Graph/'
    elif where is 'saved_model':
        return 'C:/Users/YH/PycharmProjects/AE-signal-model/result/saved_model/'
    elif where is 'yh_laptop_test_data':
        return 'C:/Users/YH/Desktop/Experiment_2018_7_13/'
    elif where is 'google_drive':
        return 'C:/Users/YH/Desktop/hooyuheng.masterWork/MASTER_PAPERWORK/temp_data_store/'


def plot_cwt_with_time_series(time_series, no_of_time_series, cwt_mat, cwt_scale,
                              maxpoint_searching_bound=24000, title='No Title'):
    '''
    Usage Recommendation:

    Usage_1 [1x time series + 1x cwt] ----
    When we just wan to aligned the time series signal with its CWT

    Usage_2 [2x time series + 1x cwt_xcor] ----
    When we are doing correlation of cwt of the 2 time series signals, we wan to aligned and
    inspect the 2 time series signal, meanwhile also display the correlation of the CWT
    -----------------------------------------------------------------------------------------------
    :param time_series: a 1d array (or) a list of 2 1d array
    :param no_of_time_series: 1 -> Usage_1 , 2 -> Usage_2
    :param cwt_mat: a 2d array
    :param cwt_scale: the scale passed into CWT function, with values in ascending order
    :param maxpoint_searching_bound: (applies to usgae 2 only) the horizontal bound on the xcor where max point is found
                                     it is recommended to be 24000 by weitang
    :return: a figure of time series and cwt heatmap
    '''

    fig = plt.figure(figsize=(10, 7.5))  # ori(10, 7.5)
    fig.suptitle(title, fontweight='bold')

    if no_of_time_series is 1:
        # config the position
        ax1 = fig.add_axes([0.1, 0.8, 0.8, 0.1])
        cwt_ax = fig.add_axes([0.1, 0.2, 0.8, 0.5], sharex=ax1)
        colorbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.01])

        # set title
        ax1.set_title('Signal in Time')
        cwt_ax.set_title('CWT of Signal')

        # plot time series
        ax1.plot(time_series)

    elif no_of_time_series is 2:
        ax1 = fig.add_axes([0.1, 0.6, 0.8, 0.1])  # lower
        ax2 = fig.add_axes([0.1, 0.8, 0.8, 0.1], sharex=ax1)  # upper
        cwt_ax = fig.add_axes([0.1, 0.2, 0.8, 0.3])
        colorbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.01])

        # set title
        ax1.set_title('Signal 2 in Time')
        ax2.set_title('Signal 1 in Time')
        cwt_ax.set_title('XCOR of CWT of Signal 1 and 2')

        # plot time series
        ax1.plot(time_series[1])
        ax2.plot(time_series[0])

        # plot middle line
        mid = cwt_mat.shape[1] // 2 + 1
        cwt_ax.axvline(x=mid, linestyle='dotted')

        # set bound for max point searching in xcor map
        upper_xcor_bound = mid + maxpoint_searching_bound
        # upper_xcor_bound = mid
        lower_xcor_bound = mid - maxpoint_searching_bound

        # plot max point
        for row_no in range(cwt_mat.shape[0]):
            max_along_x = np.argmax(cwt_mat[row_no, lower_xcor_bound:upper_xcor_bound])
            # the y-coord of the max point is set to use the real cwt scale value because setting the extent
            # in imshow has forced us to use the real value scale value and not index
            cwt_ax.scatter(max_along_x + lower_xcor_bound, cwt_scale[row_no], s=70, c='black', marker='x')

        cwt_ax.set_xlim(left=lower_xcor_bound, right=upper_xcor_bound)
        # max = np.unravel_index(np.argmax(cwt_mat, axis=None), cwt_mat.shape)
        # cwt_ax.scatter(max[1], cwt_scale[max[0]], s=70, c='black', marker='x')

    else:
        raise ValueError('param: no_of_time_series can only be 1 or 2')

    # plot heatmap
    ix = cwt_ax.imshow(cwt_mat, cmap='seismic', aspect='auto',
                       extent=[0, cwt_mat.shape[1], cwt_scale[-1], cwt_scale[0]])
    plt.colorbar(ix, cax=colorbar_ax, orientation='horizontal')
    cwt_ax.grid(linestyle='dotted')
    cwt_ax.set_xlabel('Xcor step')
    cwt_ax.set_ylabel('Scale')

    return fig


def shuffle_in_unison(a, b):
    '''
    Shuffle 2 np array with the same pattern
    :param a:
    :param b:
    :return:
    '''
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def scatter_plot(dataset, label, num_classes, feature_to_plot, annotate_all_point=True, title='No Title',
                 save_data_to_csv=False):
    '''
    This plot is recommended for usage on PCA and T-sne data, from 2d to 3d

    Drawbacks:
    3d visualizing lags when the points exceed 1000, use scatter_plot_3d_vispy()

    :param dataset: a 2d np array, where shape[0] -> sample size, shape[1] -> no of features
    :param label: label is a 1d np array (not in one hot encoding), which must aligned with dataset
    :param num_classes: num of labels present
    :param feature_to_plot: tuple of 2 or 3, the shape[1] of the dataset to be plotted. e.g. (0, 1), (0, 2, 3)
    :param annotate_all_point: label every points with classes index.
    :param title: big title on the scatter plot
    :return: a fig object
    '''
    # transfer data from array to pandas df
    dim_red_n_label = np.concatenate((dataset, label.reshape((-1, 1))), axis=1)
    dl_df = pd.DataFrame(dim_red_n_label, columns=['c_{}'.format(i) for i in range(dataset.shape[1])] + ['label'])

    # saving
    if save_data_to_csv:
        save_filename = direct_to_dir(where='result') + title + '.csv'
        dl_df.to_csv(save_filename)
        print('Scatter plot Saved --> {}'.format(save_filename))

    # create class label
    legend_label = ['class[{}m]'.format(i) for i in range(11)]

    # config color scheme for scatter plot
    cmap = cm.get_cmap('rainbow')

    # create fig
    fig = plt.figure(figsize=(10, 7.5))
    fig.suptitle(title)
    # for 2d plot
    if len(feature_to_plot) is 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('component_{}'.format(feature_to_plot[0]))
        ax.set_ylabel('component_{}'.format(feature_to_plot[1]))
        # for all classes
        for i in range(num_classes):
            # pick all rows with label == i
            one_class_vec = dl_df.loc[dl_df['label'] == i].values
            ax.scatter(x=one_class_vec[:, feature_to_plot[0]],
                       y=one_class_vec[:, feature_to_plot[1]],
                       c=cmap((i+1)/num_classes),
                       cmap=cm.rainbow,
                       label=legend_label[i])
            if annotate_all_point:
                # for each point in a class
                for j in range(one_class_vec.shape[0]):
                    plt.annotate(i, (one_class_vec[j, feature_to_plot[0]], one_class_vec[j, feature_to_plot[1]]))
    # for 3d plot
    elif len(feature_to_plot) is 3:
        ax = Axes3D(fig)
        ax.set_xlabel('component_{}'.format(feature_to_plot[0]))
        ax.set_ylabel('component_{}'.format(feature_to_plot[1]))
        ax.set_zlabel('component_{}'.format(feature_to_plot[2]))
        # scatter class by class
        for i in range(num_classes):
            # pick all rows with label == i
            one_class_vec = dl_df.loc[dl_df['label'] == i].values
            ax.scatter(xs=one_class_vec[:, feature_to_plot[0]],
                       ys=one_class_vec[:, feature_to_plot[1]],
                       zs=one_class_vec[:, feature_to_plot[2]],
                       c=cmap((i + 1) / num_classes),
                       cmap=cm.rainbow,
                       label=legend_label[i])
            if annotate_all_point:
                # for each point in a class
                for j in range(one_class_vec.shape[0]):
                    ax.text(one_class_vec[j, 0], one_class_vec[j, 1], one_class_vec[j, 2], i)

    return fig


def scatter_plot_3d_vispy(dataset, label):
    '''
    code modified from https://stackoverflow.com/questions/44766591/vispy-two-data-sets-on-same-plot-with-colors
    :param dataset: 2d ndarray, where shape[0] -> sample size, shape[1] -> no of features(i.e. mz be 3**)
    :param label: a 1d ndarray, non-one-hot-encoded
    :return: ntg, jz plot

    e.g. if the tsne has to be read from a csv file, where first 3 cols are the coordinate, last col is the label,
    use the quick code suggested below:

    tsne_filename = direct_to_dir(where='result') + 'xxx.csv'
    tsne_df = pd.read_csv(tsne_filename, index_col=0)
    dataset = tsne_df.values[:, :3]
    label = tsne_df.values[:, -1]
    scatter_plot_3d_vispy(dataset=dataset, label=label)

    '''

    # config color scheme for scatter plot
    cmap = cm.get_cmap('rainbow')

    # convert the label to RGBA
    scaler = MinMaxScaler(feature_range=(0, 1))
    label_normalized = scaler.fit_transform(label.reshape((-1, 1)).astype('float64'))
    color_label_rgba = np.array([cmap(i) for i in label_normalized.ravel()])

    # vispy code -------------------------------------------------------------------------------------------------------
    # build your visuals, that's all
    Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)

    # The real-things : plot using scene
    # build canvas
    canvas = scene.SceneCanvas(keys='interactive', show=True)

    # Add a ViewBox to let the user zoom/rotate
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = 500

    # plot ! note the parent parameter
    p1 = Scatter3D(parent=view.scene)
    p1.set_gl_state('translucent', blend=True, depth_test=True)
    p1.set_data(dataset, face_color=color_label_rgba, symbol='o', size=10,
                edge_width=0.5, edge_color='blue')

    # run
    app.run()


def lollipop_plot(x, y, test_point=None, label=None):
    '''
    This is now for plotting 2 sets of data (x, y) and (x', y')
    :param x: a list of 1d array, where shape[0]=2, shape[1]=data len
    :param y: a list of 1d array, where shape[0]=2, shape[1]=data len
    :param test_point: just an extra points, as a marker
    :param label: re
    :return:
    '''

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    markerline, stemlines, baseline = ax.stem(x[0], y[0], '-', label=label[0])
    markerline2, stemlines2, baseline2 = ax.stem(x[1], y[1], '-', label=label[1])

    # set style
    plt.setp(markerline, markerfacecolor='b')
    plt.setp(stemlines, color='b', linewidth=1, linestyle='dotted')
    plt.setp(baseline, visible=False)

    plt.setp(markerline2, markerfacecolor='r', markeredgecolor='r')
    plt.setp(stemlines2, color='r', linewidth=1, linestyle='dotted')
    plt.setp(baseline2, visible=False)

    if test_point is not None:
        markerline3, stemlines3, baseline3 = ax.stem(test_point, [1] * len(test_point), '-', label='test')
        plt.setp(markerline3, markerfacecolor='g', markeredgecolor='r')
        plt.setp(stemlines3, color='g', linewidth=1, linestyle='dotted')
        plt.setp(baseline3, visible=False)

    ax.grid(linestyle='dotted')
    ax.legend()

    return fig


def detect_close_value(x1, x2, threshold1, threshold2):
    '''
    Recommended usage:
    It is usually to process the peaks list by peakutils.indexes() from 2 sensors.

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









