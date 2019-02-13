import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from src.experiment_dataset.dataset_experiment_2019_1_3 import AcousticEmissionDataSet
from src.model_bank.dataset_2018_7_13_lcp_recognition_model import *
from src.utils.helpers import *


# ------------------------------------------------------------------------------------------------------------ DATA PREP
ae_data = AcousticEmissionDataSet(drive='G')
train_x, train_y, test_x, test_y = ae_data.random_leak_noleak_downsampled_2_include_unseen(train_split=0.8)

# train_x_reshape = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
# test_x_reshape = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
#
# train_y_cat = to_categorical(train_y, num_classes=2)
# test_y_cat = to_categorical(test_y, num_classes=2)

# ------------------------------------------------------------------------------------------------------------------ SVM
model = svm.SVC(kernel='linear', C=3, gamma=1)

# training time
time_train_start = time.time()
model.fit(train_x, train_y)
time_taken_train = time.time() - time_train_start

score = model.score(train_x, train_y)

predict = model.predict(test_x)
best_val_acc = accuracy_score(y_pred=predict, y_true=test_y)

_, _, _, _, result = compute_recall_precision_multiclass(y_pred=predict, y_true=test_y, all_class_label=[0, 1])

print('Accuracy: {:.4f}'.format(best_val_acc))
print(result)
