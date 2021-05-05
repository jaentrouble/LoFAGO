import argparse
import agent_assets.agent_models as am
from agent_assets import tools
import gym, gym_lostark
from pathlib import Path
import tqdm
import numpy as np
import tensorflow as tf

gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

TARGETS = (
    (10,6,2),
    (10,6,4),
    (9,8,2),
    (9,8,4),
    (9,6,2),
    (9,6,4),
    (9,5,2),
    (9,5,4),
    (8,8,2),
    (8,8,4),
    (8,7,2),
    (8,7,4),
    (7,7,2),
    (7,7,4),
    (6,6,4),
    (6,6,2)
)
ENV_BATCH = 1024

parser = argparse.ArgumentParser()
parser.add_argument('-n',help='Number of eval steps',type=int, dest='num')
parser.add_argument('-l','--load',help='model directory', dest='load')
args = parser.parse_args()

envs = []
for _ in range(ENV_BATCH):
    env = gym.make('AbilityStone-v0')
    envs.append(env)

model_f = am.classic_dense_vmpo_discrete

actor, critic = model_f({'obs':env.observation_space}, env.action_space)
save_dir = Path(args.load)
actor.load_weights(str(save_dir/'actor'))

obs_range = (env.observation_space.high-
             env.observation_space.low)
obs_middle = (env.observation_space.high+
              env.observation_space.low)/2

def pre_processing(obs):
    return (2*(obs-obs_middle)/obs_range)

@tf.function
def act(obs):
    a_logit = actor(pre_processing(obs))
    return tf.argmax(a_logit,axis=-1)


target_tqdm =tqdm.tqdm(TARGETS)
for TARGET in target_tqdm:
    target_tqdm.set_description(str(TARGET))
    count_t = tqdm.tqdm(total=args.num, dynamic_ncols=True)
    results = []
    done_count = 0
    o_list = []
    for env in envs:
        o_list.append(env.reset(TARGET))
    while done_count<args.num:
        o_batch = np.array(o_list)
        a_tf = act(o_batch)
        a = a_tf.numpy()

        o_list = []
        for env, a in zip(envs, a):
            o, r, done, i = env.step(a)
            if done or r>0:
                results.append(i)
                o = env.reset(TARGET)
                done_count += 1
                count_t.update()
            o_list.append(o)
    np.savetxt(str(save_dir/f'eval_{TARGET}_{args.num}.csv'),
                np.array(results),delimiter=',')
    count_t.close()