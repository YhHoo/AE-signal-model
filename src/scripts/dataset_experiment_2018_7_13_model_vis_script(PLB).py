import numpy as np
# self defined library
from src.utils.helpers import model_loader, get_activations, display_activations, break_into_train_test, \
                              reshape_3d_to_4d_tocategorical, three_dim_visualizer
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018


# -------------------[LOADING DATA]----------------------------
data = AcousticEmissionDataSet_13_7_2018(drive='F')
dataset, label = data.plb()

# split to train test data
num_classes = 41
train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=num_classes,
                                                         train_split=0.7,
                                                         verbose=True)
# reshape to satisfy conv2d input shape
train_x, train_y, test_x, test_y = reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y,
                                                                  fourth_dim=1,
                                                                  num_classes=num_classes,
                                                                  verbose=True)

# -------------------[LOADING MODEL]----------------------------

model = model_loader(model_name='PLB_2018_7_13_Classification_CNN[33k]_take0')
model.compile(loss='categorical_crossentropy', optimizer='adam')


activation = get_activations(model, model_inputs=test_x,print_shape_only=True, layer_name='conv2d_1')

activation = np.array(activation)
print(activation.shape)

for i in range(8):
    display_activations(activation[0, 0, :, :, i])

