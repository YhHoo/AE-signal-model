from scipy.signal import gausspulse
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
# self defined library
from src.controlled_dataset.ideal_dataset import white_noise
from src.utils.dsp_tools import spectrogram_scipy, one_dim_xcor_freq_band
from src.utils.helpers import three_dim_visualizer, break_into_train_test, reshape_3d_to_4d_tocategorical, \
                              ModelLogger, ModelCheckpoint, model_multiclass_evaluate
from src.model_bank.cnn_model_bank import cnn_general_v1

# Designing Gauss pulse with different time shift, contaminated with white noise

# time setting
t = np.linspace(0, 2, 5000, endpoint=False)
# original signal creating
pulse1 = gausspulse(t-0.5, fc=50)
pulse2 = gausspulse(t-0.6, fc=50)
pulse3 = gausspulse(t-0.7, fc=50)
mix_signal_1, mix_signal_2, mix_signal_3 = [], [], []
# contaminate pulse 1
for sig1, sig2 in zip(pulse1, white_noise(time_axis=t, power=0.1)):
    mix_signal_1.append(sig1+sig2)
# contaminate pulse 2
for sig1, sig2 in zip(pulse2, white_noise(time_axis=t, power=0.1)):
    mix_signal_2.append(sig1+sig2)
# contaminate pulse 3
for sig1, sig2 in zip(pulse3, white_noise(time_axis=t, power=0.1)):
    mix_signal_3.append(sig1+sig2)


fig1 = plt.figure()
ax1 = fig1.add_subplot(3, 1, 1)
ax2 = fig1.add_subplot(3, 1, 2)
ax3 = fig1.add_subplot(3, 1, 3)
ax1.plot(t, mix_signal_1)
ax2.plot(t, mix_signal_2)
ax3.plot(t, mix_signal_3)

plt.show()

class_1, class_2 = [], []
for i in range(100):
    t, f, mat1 = spectrogram_scipy(sampled_data=mix_signal_1,
                                   fs=2500,  # because 2500 points per sec
                                   nperseg=200,
                                   noverlap=0,
                                   mode='angle',
                                   return_plot=False,
                                   verbose=False)

    _, _, mat2 = spectrogram_scipy(sampled_data=mix_signal_2,
                                   fs=2500,  # because 2500 points per sec
                                   nperseg=200,
                                   noverlap=0,
                                   mode='angle',
                                   return_plot=False,
                                   verbose=False)

    _, _, mat3 = spectrogram_scipy(sampled_data=mix_signal_3,
                                   fs=2500,  # because 2500 points per sec
                                   nperseg=200,
                                   noverlap=0,
                                   mode='angle',
                                   return_plot=False,
                                   verbose=False)
    l = np.array([mat1, mat2, mat3])
    xcor_map = one_dim_xcor_freq_band(input_mat=l, pair_list=[(0, 1), (0, 2)], verbose=False)
    # signal labelling
    class_1.append(xcor_map[0])
    class_2.append(xcor_map[1])

    # for map in xcor_map:
    #     ax3 = three_dim_visualizer(x_axis=np.arange(1, map.shape[1] + 1, 1),
    #                                y_axis=f,
    #                                zxx=map,
    #                                label=['Xcor_steps', 'Frequency', 'Correlation Score'],
    #                                output='2d',
    #                                title='Xcor Map [Mag] of GaussPulse+W.noise with Difference: -0.2s')
    #     ax3.show()
class_1 = np.array(class_1)
class_2 = np.array(class_2)
all_class = [class_1, class_2]
dataset = np.concatenate(all_class, axis=0)
label = np.array([0]*class_1.shape[0] + [1]*class_2.shape[0])

print('Data set Dim: ', dataset.shape)
print('Label Dim: ', label.shape)

# training ------------------------------
num_classes = 2
train_x, train_y, test_x, test_y = break_into_train_test(input=dataset,
                                                         label=label,
                                                         num_classes=num_classes,
                                                         train_split=0.7,
                                                         verbose=True)
# reshape to satisfy conv2d input shape
train_x, train_y, test_x, test_y = reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y,
                                                                  fourth_dim=1, num_classes=num_classes, verbose=True)

model = cnn_general_v1(input_shape=(train_x.shape[1], train_x.shape[2]), num_classes=num_classes)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_logger = ModelLogger(model, model_name='cnn_general_v1')
history = model.fit(x=train_x,
                    y=train_y,
                    batch_size=30,
                    validation_data=(test_x, test_y),
                    epochs=100,
                    verbose=1)

model_logger.learning_curve(history=history, save=True)
model_multiclass_evaluate(model, test_x=test_x, test_y=test_y)

# TRAINING SUCCESSFUL !! WAITING FOR REARRANGE THE CODE