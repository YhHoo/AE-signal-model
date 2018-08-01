import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
# self defined library
from src.utils.helpers import model_loader, get_activations, break_into_train_test, \
                              reshape_3d_to_4d_tocategorical, plot_multiple_horizontal_heatmap
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018


save_dir = 'C:/Users/YH/PycharmProjects/AE-signal-model/result/'

# -------------------[LOADING DATA]----------------------------
data = AcousticEmissionDataSet_13_7_2018(drive='F')
dataset, label = data.plb()

# split to train test data
num_classes = 41
train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=num_classes,
                                                         train_split=0.95,
                                                         verbose=True,
                                                         shuffled_each_class=False)
# reshape to satisfy conv2d input shape
_, _, test_x, _ = reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y,
                                                 fourth_dim=1,
                                                 num_classes=num_classes,
                                                 verbose=True)

# -------------------[LOADING MODEL]----------------------------

model = model_loader(model_name='PLB_2018_7_13_Classification_CNN[33k]_take0')
model.compile(loss='categorical_crossentropy', optimizer='adam')

activation = get_activations(model, model_inputs=test_x, print_shape_only=True)

print(len(activation))


sample_no = 0
# for all samples
for a in activation[0]:
    # put all fmap into 1 list
    fmap_list = [a[:, :, i] for i in range(a.shape[2])]
    label = 'Maxpooling2d_3 - Sample_no ={}, Label=[{}m]'.format(sample_no, test_y[sample_no])
    save_filename = save_dir + label
    fig = plot_multiple_horizontal_heatmap(fmap_list,
                                           title=label,
                                           subplot_title='F_map')
    fig.savefig(save_filename)
    plt.close('all')
    sample_no += 1



