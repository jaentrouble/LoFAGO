import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from . import critic_models as cm
from . import encoder_models as em
from . import actor_models as am
from . import ICM
"""
Actor-Critic agent model
Agent functions return two models:
    1. encoder_model
        This takes observation only
    2. actor_model
        This takes encoded state only
    3. critic_model
        This takes encoded state and action together

Every functions should take following two as inputs:
    1. observation_space
    2. action_space : Box expected
"""

# def unity_conv_iqn_icm(observation_space, action_space):
#     encoder_f = em.encoder_simple_conv

#     actor = am.actor_simple_dense(observation_space, action_space, encoder_f)

#     critic = cm.critic_dense_iqn(observation_space, action_space, encoder_f)

#     icm_models = ICM.ICM_dense(observation_space, action_space, encoder_f)

#     return actor, critic, icm_models

def unity_conv_vmpo(observation_space, action_space):
    encoder_f = em.encoder_simple_conv

    actor = am.actor_vmpo_dense(observation_space, action_space, encoder_f)

    critic = cm.critic_v_dense(observation_space, action_space, encoder_f)

    return actor, critic

def classic_dense_vmpo(observation_space, action_space):
    encoder_f = em.encoder_simple_dense

    actor = am.actor_vmpo_dense(observation_space, action_space, encoder_f)

    critic = cm.critic_v_dense(observation_space, action_space, encoder_f)

    return actor, critic

def classic_dense_ppo(observation_space, action_space):
    encoder_f = em.encoder_simple_dense

    actor = am.actor_ppo_dense(observation_space, action_space, encoder_f)

    critic = cm.critic_v_dense(observation_space, action_space, encoder_f)

    return actor, critic

def classic_dense_a2c(observation_space, action_space):
    encoder_f = em.encoder_simple_dense

    actor = am.actor_a2c_dense_mu(observation_space, action_space, encoder_f)

    critic = cm.critic_v_dense(observation_space, action_space, encoder_f)

    return actor, critic

def classic_mini_a2c(observation_space, action_space):
    encoder_f = em.encoder_mini_dense

    actor = am.actor_a2c_mini_mu(observation_space, action_space, encoder_f)

    critic = cm.critic_v_mini(observation_space, action_space, encoder_f)

    return actor, critic
