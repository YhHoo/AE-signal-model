import keras as kr
# self defined library
from src.controlled_dataset.ideal_dataset import gauss_pulse_timeshift_dataset
from src.utils.helpers import break_balanced_class_into_train_test, reshape_3d_to_4d_tocategorical, \
                              ModelLogger, evaluate_model_for_all_class

dataset, label = gauss_pulse_timeshift_dataset(class_sample_size=100)

# training ------------------------------
num_classes = 2
train_x, train_y, test_x, test_y = break_balanced_class_into_train_test(input=dataset,
                                                                        label=label,
                                                                        num_classes=num_classes,
                                                                        train_split=0.7,
                                                                        verbose=True)
# reshape to satisfy conv2d input shape
train_x, train_y, test_x, test_y = reshape_3d_to_4d_tocategorical(train_x, train_y, test_x, test_y,
                                                                  fourth_dim=1, num_classes=num_classes, verbose=True)

model = kr.Sequential()
model.add(kr.layers.Conv2D(filters=36, kernel_size=(2, 2), strides=(1, 1), activation='relu',
                           input_shape=(train_x.shape[1], train_x.shape[2], 1)))
model.add(kr.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(kr.layers.Conv2D(filters=59, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
model.add(kr.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(kr.layers.Flatten())

model.add(kr.layers.Dense(50, activation='relu'))
model.add(kr.layers.Dense(2, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_logger = ModelLogger(model, model_name='CNN_gausspulse_timeshift')
history = model.fit(x=train_x,
                    y=train_y,
                    batch_size=30,
                    validation_data=(test_x, test_y),
                    epochs=100,
                    verbose=1)

model_logger.learning_curve(history=history, show=True)
evaluate_model_for_all_class(model, test_x=test_x, test_y=test_y)

