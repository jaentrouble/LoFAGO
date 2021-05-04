import argparse
import agent_assets.agent_models as am
from agent_assets import tools
import gym, gym_lostark
from pathlib import Path
import tqdm
import numpy as np
import tensorflow as tf

TARGET = (0,0,10)

parser = argparse.ArgumentParser()
parser.add_argument('-n',help='Number of eval steps',type=int, dest='num')
parser.add_argument('-l','--load',help='model directory', dest='load')
args = parser.parse_args()

env = gym.make('AbilityStone-v0')
env = tools.EnvWrapper_AbilityStone(env)

model_f = am.classic_dense_vmpo_discrete

actor, critic = model_f(env.observation_space, env.action_space)
save_dir = Path(args.load)
actor.load_weights(str(save_dir/'actor'))

obs_range = (env.observation_space['obs'].high-
             env.observation_space['obs'].low)
obs_middle = (env.observation_space['obs'].high+
              env.observation_space['obs'].low)/2

def pre_processing(obs):
    return (2*(obs['obs']-obs_middle)/obs_range)[np.newaxis,...]

@tf.function
def act(obs):
    a_logit = actor(pre_processing(o))[0]
    return tf.argmax(a_logit)


results = []
for _ in tqdm.trange(args.num):
    o = env.reset(TARGET)
    done = False
    while not done:
        a_tf = act(o)
        a = a_tf.numpy()
        o, r, done, i = env.step(a)
    results.append(i)
np.savetxt(str(save_dir/f'eval_{TARGET}_{args.num}.csv'),
            np.array(results),delimiter=',')