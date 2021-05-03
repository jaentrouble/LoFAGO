import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from . import A_hparameters as hp
import numpy as np
"""
Q models take two inputs:
    1. encoded states
    2. actions
and returns an expected future reward (a single float).

V models take one input:
    1. encoded states

Critic model functions should take following arguments:
    1. observation_space
    2. action_space : Box (Even if the model is V and not need action)
    3. encoder_f
"""


def critic_v_dense(observation_space, action_space, encoder_f):
    encoded_state, encoder_inputs = encoder_f(observation_space)
    x = layers.Flatten(name='critic_flatten_state')(encoded_state)

    x = layers.Dense(256, activation='relu',
                     name='critic_dense1')(x)
    x = layers.Dense(128, activation='relu',
                     name='critic_dense2')(x)
    x = layers.Dense(64, activation='relu',
                     name='critic_dense3')(x)
    x = layers.Dense(1, activation='linear',dtype='float32',
                           name='critic_dense4')(x)
    outputs = tf.squeeze(x, name='critic_squeeze')
    outputs = layers.Activation('linear',dtype=tf.float32,
                                name='critic_float32')(outputs)

    model = keras.Model(
        inputs=encoder_inputs, 
        outputs=outputs,
        name='critic'
    )

    return model

def critic_v_mini(observation_space, action_space, encoder_f):
    encoded_state, encoder_inputs = encoder_f(observation_space)
    x = layers.Flatten(name='critic_flatten_state')(encoded_state)

    x = layers.Dense(64, activation='relu',
                     name='critic_dense1')(x)
    x = layers.Dense(1, activation='linear',dtype='float32',
                           name='critic_dense4')(x)
    outputs = tf.squeeze(x, name='critic_squeeze')
    outputs = layers.Activation('linear',dtype=tf.float32,
                                name='critic_float32')(outputs)

    model = keras.Model(
        inputs=encoder_inputs, 
        outputs=outputs,
        name='critic'
    )

    return model


def critic_v_dense_iqn(observation_space, action_space, encoder_f):
    """
    IQN model takes one more input : 
        tau : tf.Tensor
            shape (batch, IQN_SUPPORT)
    """
    encoded_state, encoder_inputs = encoder_f(observation_space)
    x = layers.Flatten(name='critic_flatten_state')(encoded_state)

    # Shape (batch, 256)
    x = layers.Dense(256, activation='relu',
                     name='critic_dense1')(x)

    # Shape (batch, support)
    tau_input = keras.Input((hp.IQN_SUPPORT,), name='tau')
    # Shape (batch, support, 1)
    tau_reshape = tau_input[...,tf.newaxis]
    pi_range = np.pi * tf.range(hp.IQN_COS_EMBED, dtype=tf.float32)
    # Shape (batch, support, embed)
    tau_cos = tf.cos(tau_reshape * pi_range, name='tau_cos')
    # Shape (batch, support, 256)
    phi = layers.Dense(256, activation='relu',
                       name='phi')(tau_cos)

    # Shape (batch, 1, 256)
    x_reshaped = layers.Reshape((1,256),name='critic_reshape_x')(x)

    # Shape (batch, support, 256)
    x = layers.Multiply(name='critic_mul_phi')([phi, x_reshaped])
    
    x = layers.Dense(512, activation='relu',
                     name='critic_dense2')(x)
    x = layers.Dense(512, activation='relu',
                     name='critic_dense3')(x)
    x = layers.Dense(512, activation='relu',
                     name='critic_dense4')(x)

    # Shape (batch, support, 1)
    x = layers.Dense(1, activation='linear',
                     dtype='float32', name='critic_dense5')(x)
    # Output shape (batch, support)
    outputs = tf.squeeze(x, name='critic_squeeze')
    outputs = layers.Activation('linear',dtype=tf.float32,
                                name='critic_float32')(outputs)

    model = keras.Model(
        inputs=[tau_input] + encoder_inputs, 
        outputs=outputs,
        name='critic'
    )

    return model

