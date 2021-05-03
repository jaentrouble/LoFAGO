import gym
import gym_lostark
import time
import numpy as np
from agent_assets.Agent import Player
import agent_assets.agent_models as am
from agent_assets import tools
from agent_assets.tools import k_steps
import agent_assets.A_hparameters as hp
from tqdm import tqdm
import argparse
from tensorflow.profiler.experimental import Profile
from datetime import timedelta
from agent_assets.replaybuffer import ReplayBuffer

ENVIRONMENT = 'mouseUnity-v0'

env_kwargs = dict(
    ip='localhost',
    port=7777
)

hp.CLASSIC = False

model_f = am.unity_conv_vmpo

hp.Actor_activation = 'tanh'

evaluate_f = tools.evaluate_unity

parser = argparse.ArgumentParser()
parser.add_argument('-r','--render', dest='render',action='store_true', default=False)
parser.add_argument('--step', dest='total_steps',default=100000, type=int)
parser.add_argument('-n','--logname', dest='log_name',default=None)
parser.add_argument('-pf', dest='profile',action='store_true',default=False)
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)
parser.add_argument('-l','--load', dest='load', default=None)
args = parser.parse_args()

total_steps = int(args.total_steps)
my_tqdm = tqdm(total=total_steps, dynamic_ncols=True)

hp.Algorithm = 'V-MPO'

hp.Batch_size = 256
hp.Buf.N = 32
hp.k_train_step = 8
hp.Q_discount = 0.99
hp.Target_update = 100

hp.Model_save = 5000
hp.histogram = 1000

hp.lr['common'].halt_steps = 0
hp.lr['common'].start = 1e-4
hp.lr['common'].end = 1e-4
hp.lr['common'].nsteps = 2e4
hp.lr['common'].epsilon = 1e-5
hp.lr['common'].grad_clip = 0.1

hp.lr['encoder'].halt_steps = 0
hp.lr['encoder'].start = 1e-5
hp.lr['encoder'].end = 1e-5
hp.lr['encoder'].nsteps = 1e6
hp.lr['encoder'].epsilon = 1e-5
hp.lr['encoder'].grad_clip = None

hp.lr['forward'] = hp.lr['encoder']
hp.lr['inverse'] = hp.lr['encoder']

hp.VMPO_eps_eta = 1e-1
hp.VMPO_eps_alpha_mu = 1e-2
hp.VMPO_eps_alpha_sig = 1e-5

hp.IQN_ENABLE = False

hp.ICM_ENABLE = False
hp.ICM_intrinsic = 1.0
hp.ICM_loss_forward_weight = 0.2



# For benchmark
st = time.time()


env = gym.make(ENVIRONMENT, **env_kwargs)
if hp.CLASSIC:
    env = tools.EnvWrapper(env)
last_obs = env.reset()
render = args.render
if render :
    env.render()

player = Player(
    observation_space= env.observation_space, 
    action_space= env.action_space, 
    model_f= model_f,
    m_dir=args.load,
    log_name= args.log_name,
    mixed_float=args.mixed_float,
)

need_to_eval = False
buf = ReplayBuffer(env.observation_space, env.action_space)
reset_buffer = True
cum_reward = 0.0
rounds = 0
per_round_steps = 0
act_steps = 0

if args.profile:
    for step in range(20):
        last_obs, cum_reward, rounds, act_steps,\
        per_round_steps, evaluated, reset_buffer\
            = k_steps(
                reset_buffer,
                buf,
                player,
                env,
                last_obs,
                my_tqdm,
                cum_reward,
                rounds,
                act_steps,
                per_round_steps,
                render,
                need_to_eval,
                hp.k_train_step,
                evaluate_f,
            )

    with Profile(f'logs/{args.log_name}'):
        for step in range(5):
            last_obs, cum_reward, rounds, act_steps,\
            per_round_steps, evaluated, reset_buffer\
                = k_steps(
                    reset_buffer,
                    buf,
                    player,
                    env,
                    last_obs,
                    my_tqdm,
                    cum_reward,
                    rounds,
                    act_steps,
                    per_round_steps,
                    render,
                    need_to_eval,
                    hp.k_train_step,
                    evaluate_f
                )
    remaining_steps = total_steps - 25
    while my_tqdm.n < total_steps:
        if ((my_tqdm.n + 25*hp.k_train_step) % hp.Model_save) \
            in range(hp.k_train_step):
            need_to_eval = True
        last_obs, cum_reward, rounds, act_steps,\
        per_round_steps, evaluated, reset_buffer\
            = k_steps(
                reset_buffer,
                buf,
                player,
                env,
                last_obs,
                my_tqdm,
                cum_reward,
                rounds,
                act_steps,
                per_round_steps,
                render,
                need_to_eval,
                hp.k_train_step,
                evaluate_f,
            )
        if evaluated:
            need_to_eval = False

else :
    while my_tqdm.n < total_steps:
        if (my_tqdm.n>0) and ((my_tqdm.n % hp.Model_save) \
            in range(hp.k_train_step)):
            need_to_eval = True
        last_obs, cum_reward, rounds, act_steps,\
        per_round_steps, evaluated, reset_buffer\
            = k_steps(
                reset_buffer,
                buf,
                player,
                env,
                last_obs,
                my_tqdm,
                cum_reward,
                rounds,
                act_steps,
                per_round_steps,
                render,
                need_to_eval,
                hp.k_train_step,
                evaluate_f,
            )
        if evaluated:
            need_to_eval = False

player.save_model()
score = evaluate_f(player, env, 'mp4')
print('eval_score:{0}'.format(score))
d = timedelta(seconds=time.time() - st)
print(f'{total_steps}steps took {d}')
my_tqdm.close()

