import tensorflow as tf
from tensorflow import math as tm
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import agent_assets.A_hparameters as hp

class QModel(keras.Model):
    def train_step(self, data):
        o, r, d, a, target_q = data
        num_actions = target_q.shape[-1]
        q_samp = r + tf.cast(tm.logical_not(d), tf.float32) * \
                     hp.Q_discount * \
                     tm.reduce_max(target_q, axis=1)
        mask = tf.one_hot(a, num_actions, dtype=tf.float32)

        with tf.GradientTape() as tape:
            q = self(o, training=True)
            q_sa = tf.math.reduce_sum(q*mask, axis=1)
            loss = keras.losses.MSE(q_samp, q_sa)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(q_sa, q_samp)
        return {m.name: m.result() for m in self.metrics}