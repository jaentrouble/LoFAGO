import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
"""
ICM functions should take following arguments:
    1. observation_space
    2. action_space
    3. encoder_f

ICM functions returns 3 models:
    1. Encoder
    2. Forward Model
    3. Inverse Model
"""

def ICM_dense(observation_space, action_space, encoder_f):
    # Encoder model
    encoded_state, encoder_inputs = encoder_f(observation_space)
    encoder_model = keras.Model(
        inputs=encoder_inputs,
        outputs = encoded_state,
        name='ICM_encoder'
    )

    # Inverse Model
    feature_s_input = keras.Input(
        encoded_state.shape[1:], 
        name='s_feature'
    )
    feature_sp_input = keras.Input(
        encoded_state.shape[1:], 
        name='sp_feature'
    )
    flattened_s = layers.Flatten(name='inverse_s_flatten')(
                                    feature_s_input)
    flattened_sp = layers.Flatten(name='inverse_sp_flatten')(
                                    feature_sp_input)
    concated_feature = layers.Concatenate(name='inverse_concat')(
        [flattened_s, flattened_sp]
    )
    action_shape = action_space.shape
    action_num = tf.math.reduce_prod(action_shape)
    i_x = layers.Dense(256, activation='relu',
                       name='inverse_Dense1')(concated_feature)
    i_x = layers.Dense(action_num, name='inverse_Dense2')(i_x)
    i_x = layers.Reshape(action_space.shape, name='inverse_reshape')(i_x)
    inverse_outputs = layers.Activation('linear', dtype='float32')(i_x)
    inverse_model = keras.Model(
        inputs=[feature_s_input, feature_sp_input],
        outputs=inverse_outputs,
        name='ICM_inverse_model'
    )

    # Forward Model
    action_input = keras.Input(
        action_space.shape,
        name='action'
    )
    feature_s_input = keras.Input(
        encoded_state.shape[1:], 
        name='s_feature'
    )
    flattened_a = layers.Flatten(name='forward_a_flatten')(
                                    action_input)
    flattened_s = layers.Flatten(name='forward_s_flatten')(
                                    feature_s_input)
    feature_num = flattened_s.shape[-1]
    concated_inputs = layers.Concatenate(name='forward_concat')(
        [flattened_a, flattened_s]
    )
    f_x = layers.Dense(256, activation='relu',
                        name='forward_Dense1')(concated_inputs)
    f_x = layers.Dense(256, activation='relu',
                        name='forward_Dense2')(f_x)
    f_x = layers.Dense(feature_num, name='forward_Dense3')(f_x)
    f_x = layers.Reshape(encoded_state.shape[1:],
                         name='forward_reshape')(f_x)
    forward_outputs = layers.Activation('linear',dtype='float32')(f_x)
    forward_model = keras.Model(
        inputs=[action_input, feature_s_input],
        outputs=forward_outputs,
        name='ICM_forward_model',
    )

    return encoder_model, inverse_model, forward_model