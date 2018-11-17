'''
THIS SCRIPT IS TO FEED TRAINED MODEL WITH TEST DATASET AND FIND THE F1-SCORE
'''

import time
import matplotlib.patches as mpatches
import tensorflow as tf
from src.utils.helpers import *
from src.experiment_dataset.dataset_experiment_2018_10_3 import AcousticEmissionDataSet_3_10_2018

# instruct GPU to allocate only sufficient memory for this script
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# loading dataset
ae_data = AcousticEmissionDataSet_3_10_2018(drive='F')
train_x, train_y, test_x, test_y = ae_data.lcp_by_distance_dataset_multi_class(train_split=0.7)

train_x_reshape = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
test_x_reshape = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
train_y_cat = to_categorical(train_y, num_classes=6)
test_y_cat = to_categorical(test_y, num_classes=6)

# loading best model saved
lcp_best_model = load_model(model_name='LCP_Dist_Recog_2')
print(lcp_best_model.summary())

# test with val data
time_predict_start = time.time()
prediction = lcp_best_model.predict(test_x_reshape)
time_predict = time.time() - time_predict_start

prediction_argmax = np.argmax(prediction, axis=1)
actual_argmax = np.argmax(test_y_cat, axis=1)
print(prediction_argmax)
print(actual_argmax)


compute_recall_precision_multiclass(y_pred=prediction_argmax, y_true=actual_argmax, all_class_label=[0, 1, 2, 3, 4, 5])





