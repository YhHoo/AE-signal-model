import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018
from src.model_bank.dataset_2018_7_13_lcp_recognition_model import lcp_recognition_binary_model, \
    lcp_recognition_binary_model_2
from src.utils.helpers import *

ae_data = AcousticEmissionDataSet_13_7_2018(drive='F')
train_x, train_y, test_x, test_y = ae_data.lcp_recognition_binary_class_dataset(train_split=0.6)

train_x_reshape = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
test_x_reshape = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

lcp_model = lcp_recognition_binary_model()
lcp_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# saving setting
logger = ModelLogger(model=lcp_model, model_name='LCP_Recog_1')
save_weight_checkpoint = logger.save_best_weight_cheakpoint(monitor='val_loss', period=5)

# start training
history = lcp_model.fit(x=train_x_reshape,
                        y=train_y,
                        validation_data=(test_x_reshape, test_y),
                        callbacks=[save_weight_checkpoint],
                        epochs=100,
                        batch_size=100,
                        shuffle=True,
                        verbose=2)

logger.learning_curve(history=history, show=True)
logger.save_architecture(save_readable=True)


# evaluate
prediction = lcp_model.predict(test_x_reshape)
plt.plot(test_y, color='r', label='Actual')
plt.plot(prediction, color='b', label='Prediction', linestyle='None', marker='x')
plt.title('Classifier Output in Visual')
plt.legend()
plt.show()

# fi-score


