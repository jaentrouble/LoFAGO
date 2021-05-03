import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def nc(name, postfix):
    """nc (name check)
    If name is None, return None
    Else, append name and postfix
    """
    if name is None:
        return None
    else:
        return '_'.join([name,postfix])

"""
Usefool tools to make networks

Every functions should:
    1. 'inputs' argument as the first argument,
    2. return outputs tensor
So that every functions should be used in this format:
    output = function(inputs, *args, **kwargs)
"""
def res_block(inputs, inner_layers, name=None):
    """
    A simple res block without strides
    Number of filters will be the same as inputs
    Every kernel size is 3x3

    Arguments
    ---------
    inputs : keras.Input or tf.Tensor
        input tensor
    inner_layers : int
        number of layers to skip connection
    name : str
        name of this block
    """
    filters = inputs.shape[-1]
    x = inputs
    for i in range(inner_layers):
        x = layers.Conv2D(
            filters,
            3,
            padding='same',
            activation='relu',
            name=nc(name, f'conv{i}')
        )(x)
    added = layers.Add(name=nc(name,'add_res'))([inputs,x])
    outputs = layers.ReLU(name=nc(name,'relu'))(added)
    return outputs
    

