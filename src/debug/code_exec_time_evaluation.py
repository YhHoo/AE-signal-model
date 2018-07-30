'''
This code is for us to evaluate how fast our code finish execution
Just update the param setup to the code that run only once e.g. import lib, const initialize,
np_code, sp_code which represent the 2 codes that are to be compared in exec time.
'''
from timeit import timeit

# your only-run-once code
setup = '''
import numpy as np
from src.utils.helpers import model_loader, model_multiclass_evaluate, break_into_train_test, reshape_3d_to_4d_tocategorical
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018

data = AcousticEmissionDataSet_13_7_2018(drive='F')
dataset, label = data.plb()

# split to train test data
num_classes = 41
train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=num_classes,
                                                         train_split=0.7,
                                                         verbose=True)
print(test_y)
# reshape to satisfy conv2d input shape
train_x, train_y, test_x, test_y = reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y,
                                                                  fourth_dim=1,
                                                                  num_classes=num_classes,
                                                                  verbose=True)
                                                                  
model = model_loader(model_name='PLB_2018_7_13_Classification_CNN[33k]_take0')
model.compile(loss='categorical_crossentropy', optimizer='adam')
'''

# your code that is to be compared
np_code_title = 'Predicting all Test Data'
sp_code_title = None
np_code = 'prediction = model.predict(test_x)'
# sp_code = 'correlate_scipy(l, m, \'full\', method=\'fft\')'

# result displaying
print('-----------[Code execution speed test]-------------')
print('{} exec time: '.format(np_code_title), timeit(setup=setup, stmt=np_code, number=1))
# print('{} exec time: '.format(sp_code_title), timeit(setup=setup, stmt=sp_code, number=1))
