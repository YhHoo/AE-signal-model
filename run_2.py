'''
To test my CNN model with MNIST dataset, because if it works, it shud works for AE dataset
Result: Works
'''
import numpy as np
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
# self declared library
from ideal_dataset import noise_time_shift_dataset
from utils import break_into_train_test, ModelLogger, model_multiclass_evaluate, reshape_3d_to_4d_tocategorical
from cnn_model_bank import cnn_28_28_mnist_10class


# Plot ad hoc mnist instances
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('Train_x dim: ', X_train.shape)
print('Train_y dim: ', y_train.shape)
print('Test_x dim: ', X_test.shape)
print('Test_y dim: ', y_test.shape)


# plot 4 images as gray scale
def show_train_data():
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


X_train = X_train / 255
X_test = X_test / 255

train_x, train_y, test_x, test_y = reshape_3d_to_4d_tocategorical(X_train, y_train, X_test, y_test,
                                                                  num_classes=10,
                                                                  verbose=True)

model = cnn_28_28_mnist_10class()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# logging
model_logger = ModelLogger(model, model_name='CNN_MNIST_28_28')
# train
history = model.fit(x=train_x,
                    y=train_y,
                    validation_data=(test_x, test_y),
                    epochs=10,
                    batch_size=200,
                    verbose=1)
model_logger.learning_curve(history=history, show=True, title='CNN_MNIST_28_28')

model_multiclass_evaluate(model, test_x=test_x, test_y=test_y)




