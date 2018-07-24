'''
This module is to customize own loss, activation, layer,
'''
import keras.backend as K


def categorical_crossentropy_yh(y_true, y_pred):
    '''
    this loss is actually categorical_crossentropy taken directly from keras.losses
    except the from_logits is set True
    '''
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)