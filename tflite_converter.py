import tensorflow as tf
import agent_assets.agent_models as am
from agent_assets import tools
from pathlib import Path
import gym, gym_lostark

MODEL_PATH = Path('savefiles/stone_1/7')

env = gym.make('AbilityStone-v0')

model_f = am.classic_dense_vmpo_discrete
actor, critic = model_f({'obs':env.observation_space}, env.action_space)
actor.load_weights(str(MODEL_PATH/'actor'))

actor.save('tmp_stone_model')

converter = tf.lite.TFLiteConverter.from_saved_model('tmp_stone_model')
tflite_model = converter.convert()

with open(MODEL_PATH/'tflite_model.tflite','wb') as f:
    f.write(tflite_model)