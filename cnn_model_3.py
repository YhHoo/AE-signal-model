# ------------------------------------------------------
# this model expecting the input data of 0.2 seconds with shape of (3000, 40) only.
# ------------------------------------------------------

from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential