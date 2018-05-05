from progressbar import ProgressBar, Percentage, Bar, SimpleProgress, ETA


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
