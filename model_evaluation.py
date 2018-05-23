from keras.utils import to_categorical
# self defined library
from utils import model_loader, model_multiclass_evaluate
from dataset_experiment_16_5_2018 import AccousticEmissionDataSet_16_5_2018

# -------------------[LOADING DATA]----------------------------
# data set
ae_data = AccousticEmissionDataSet_16_5_2018()
train_x, train_y, test_x, test_y = ae_data.sleak_1bar_7pos(f_range=(0, 700))

# reshape to satisfy conv2d input shape
train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))

# convert
train_y = to_categorical(train_y, num_classes=7)
test_y = to_categorical(test_y, num_classes=7)


# data summary
print('\n----------INPUT DATA DIMENSION---------')
print('Train_X dim: ', train_x.shape)
print('Train_Y dim: ', train_y.shape)
print('Test_X dim: ', test_x.shape)
print('Test_Y dim: ', test_y.shape)


# -------------------[LOADING MODEL]----------------------------

model = model_loader(model_name='test2_CNN_23_5_18',
                     dir='result/23-5-18/')
model.compile(loss='categorical_crossentropy', optimizer='adam')

print('------------MODEL EVALUATION-------------')
model_multiclass_evaluate(model, test_x=test_x, test_y=test_y)


