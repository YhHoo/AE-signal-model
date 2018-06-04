# To test my CNN model with MNIST dataset, because if it works, it shud works for AE dataset

import numpy as np
from keras.utils import to_categorical
# self declared library
from ideal_dataset import noise_time_shift_dataset
from utils import break_into_train_test, ModelLogger, model_multiclass_evaluate, reshape_3d_to_4d_tocategorical
from cnn_model_bank import cnn_2_51_3class_v1

