from progressbar import ProgressBar, Percentage, Bar, SimpleProgress, ETA
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import model_from_json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.mplot3d import Axes3D  # do not delete
import numpy as np
from nptdms import TdmsFile
from os import listdir


class ProgressBarForLoop:
    '''
    Try out more progressbar from https://github.com/coagulant/progressbar-python3/blob/master/examples.py
    progress bar, maxval is like max value in a ruler, and set the progress with update()
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
        self.path = 'C:/Users/YH/PycharmProjects/AE-signal-model/src/scripts/saved_model/' + model_name

    def save_architecture(self, save_readable=False):
        # serialize and saving the model structure to JSON
        model_json = self.model.to_json()
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
        fig = plt.figure(figsize=(6, 4))
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='test_loss')
        plt.plot(history.history['acc'], label='train_acc')
        plt.plot(history.history['val_acc'], label='test_acc')
        plt.legend()
        plt.grid()
        plt.title(title)
        if save:
            plt.savefig(self.path + '.png')
        if show:
            plt.show()
        # free up memory
        plt.close()


def model_loader(model_name=None):
    '''
    :param model_name: The model name
    :param dir: The location that contains .h5, .json of the model
    :return: a model loaded with .h5 and .json
    AIM: this just simplifies the model loading procedure by wrapping them in one.
    This has to be followed by model.compile() if we wish to train the model later
    '''
    dir = 'C:/Users/YH/PycharmProjects/AE-signal-model/src/scripts/saved_model/'
    path = dir + model_name

    # load architecture from .json
    with open(path + '.json', 'r') as f:
        model = model_from_json(f.read())

    # load weights from .h5
    model.load_weights(path + '.h5')
    print('Model Loaded !')

    return model


def model_multiclass_evaluate(model, test_x, test_y):
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


def break_into_train_test(input, label, num_classes, shuffled_each_class=True, train_split=0.7, verbose=False):
    '''
    :param input: expect a 3d np array where 1st index is total sample size
    :param label: expect a 1d np array of same size as input.shape[0]
    :param num_classes: total classes to break into
    :param shuffled_each_class: it will shuffle the samples in every class
    :param verbose: print the summary of train test size
    :return: a train and test set

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


def three_dim_visualizer(x_axis, y_axis, zxx, label, output, title='None', vis_range=[None, None, None, None]):
    '''
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


def multiplot_timeseries(input, subplot_titles, main_title):
    '''
    Aspect axis[0] of input is no. of sensors/diff features, axis[1] is time steps
    :param input: a 2d array
    :param subplot_titles: title for every plot
    :param main_title: the big title
    :return: rectangular fig obj
    '''
    no_of_plot = input.shape[0]
    fig = plt.figure(figsize=(5, 8))
    fig.suptitle(main_title, fontweight="bold")
    fig.subplots_adjust(hspace=0.7, top=0.9, bottom=0.03)
    # first plot
    ax1 = fig.add_subplot(no_of_plot, 1, 1)
    ax1.plot(input[0])
    ax1.set_title(subplot_titles[0], size=8)
    # the rest of the plot
    for i in range(1, no_of_plot, 1):
        ax = fig.add_subplot(no_of_plot, 1, i+1, sharex=ax1)  # add in sharey=ax1 if wan to share y axis too
        ax.plot(input[i])
        ax.set_title(subplot_titles[i], size=8)

    return fig


def dual_heatmap_plot(input_1, input_2):
    '''
    This function plot 2 heatmap (e.g. xcor map, FT map) in one fig, for comparison.
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



