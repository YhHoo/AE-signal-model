'''
To test my CNN model with MNIST dataset, because if it works, it shud works for AE dataset
'''
import numpy as np
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
# self declared library
from ideal_dataset import noise_time_shift_dataset
from utils import break_into_train_test, ModelLogger, model_multiclass_evaluate, reshape_3d_to_4d_tocategorical
from cnn_model_bank import cnn_2_51_3class_v1


# Plot ad hoc mnist instances
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


def show_train_data():
    # plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(X_train[10], cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(X_train[11], cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(X_train[12], cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(X_train[13], cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()
