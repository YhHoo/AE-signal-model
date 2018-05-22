from progressbar import ProgressBar, Percentage, Bar, SimpleProgress, ETA
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model


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
            path = self.path + 'txt'
            with open(path, 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))

    def save_best_weight_cheakpoint(self, monitor='val_loss', mode='min', period=1):
        '''
        :param monitor: value to monitor when saving
        :param mode: what value should be save
        :param period: how frequent to check and save
        :return: a callback_list to be placed on fit(callbacks=...)
        '''
        path = self.path + '.h5'
        checkpoint = ModelCheckpoint(filepath=path,
                                     monitor=monitor,
                                     verbose=1,
                                     save_best_only=True,
                                     mode=mode,  # for acc, it should b 'max'; for loss, 'min'
                                     period=period)  # no of epoch btw checkpoints
        callback_list = [checkpoint]
        return callback_list

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





