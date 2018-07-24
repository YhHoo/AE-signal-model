'''
This module is to customize own loss, activation, layer,
'''
from __future__ import absolute_import
import keras.backend as K
import six
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import serialize_keras_object


def categorical_crossentropy_yh(y_true, y_pred):
    '''
    this loss is actually categorical_crossentropy taken directly from keras.losses
    except the from_logits is set True
    '''
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def serialize(loss):
    return serialize_keras_object(loss)


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)

    