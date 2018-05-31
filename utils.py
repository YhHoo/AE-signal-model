from progressbar import ProgressBar, Percentage, Bar, SimpleProgress, ETA
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np


# Try out more progressbar from https://github.com/coagulant/progressbar-python3/blob/master/examples.py
# progress bar, maxval is like max value in a ruler, and set the progress with update()
class ProgressBarForLoop:
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
    '''
    def __init__(self, model, model_name):
        self.model = model
        self.path = 'result/' + model_name

    def save_architecture(self, save_readable=True):
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
        callback_list = [checkpoint]

        return callback_list

    # this function use the model history returned by fit() to plot learning curve and save it
    def learning_curve(self, history, save=False, show=False, title='default'):
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='test_loss')
        plt.legend()
        plt.title(title)
        if save:
            plt.savefig(self.path + '.png')
        if show:
            plt.show()
        # free up memory
        plt.close()


def model_loader(model_name=None, dir=None):
    '''
    :param model_name: The model name
    :param dir: The location that contains .h5, .json of the model
    :return: a model loaded with .h5 and .json
    AIM: this just simplifies the model loading procedure by wrapping them in one.
    This has to be followed by model.compile() if we wish to train the model later
    '''
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
    # e.g. [0 1 0 0] --> 1
    prediction = model.predict(test_x)
    prediction = np.argmax(prediction, axis=1)
    actual = np.argmax(test_y, axis=1)

    # visualize the multiclass classification accuracy
    plt.plot(actual, color='r', label='Actual')
    plt.plot(prediction, color='b', label='Prediction')
    plt.title('Multiclassifer Accuracy Visualization')
    plt.legend()
    plt.show()


def break_into_train_test(input, label, num_classes, train_split=0.7, verbose=False):
    '''
    :param input: expect a 3d np array where 1st index is total sample size
    :param label: expect a 1d np array of same size as input.shape[0]
    :param num_classes: total classes to break into
    :param verbose: print the summary of train test size
    :return: a train and test set
    AIM----------------------------------
    This is when we receive a list of N classes samples all concatenate together sequentially
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
    print('\n----------TRAIN AND TEST SET---------')
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
        print('Split Index from start: ', split_index_from_start)
        print('Train_x Dim: ', train_x.shape)
        print('Test_x Dim: ', test_x.shape)
        print('Train_y Dim:', train_y.shape)
        print('Test_y Dim:', test_y.shape)

    # return
    return train_x, train_y, test_x, test_y

