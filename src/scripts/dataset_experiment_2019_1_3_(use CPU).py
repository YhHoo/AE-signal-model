from src.utils.helpers import *
from src.utils.dsp_tools import *
from sklearn.model_selection import train_test_split
import time

data = np.random.random((20000, 2000))
label = np.concatenate((np.ones(10000), np.zeros(10000)), axis=0)
label = label[np.random.permutation(len(label))]
print(data)
print(label)

train_x, test_x, train_y, test_y = train_test_split(data,
                                                    label,
                                                    train_size=0.7,
                                                    shuffle=True)
print(train_x.shape)
print(train_y.shape)


train_x_reshape = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
test_x_reshape = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

train_y_cat = to_categorical(train_y, num_classes=2)
test_y_cat = to_categorical(test_y, num_classes=2)

# MODEL LOADING
lcp_model = load_model(model_name='LNL_44x6')
lcp_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
print(lcp_model.summary())

total_epoch = 10
time_train_start = time.time()
history = lcp_model.fit(x=train_x_reshape,
                        y=train_y_cat,
                        validation_data=(test_x_reshape, test_y_cat),
                        epochs=total_epoch,
                        batch_size=200,
                        shuffle=True,
                        verbose=2)
time_train = time.time() - time_train_start
