import gym
import gym_mouse
import time
import agent_assets.agent_models as am
from agent_assets.parallel_tools import ParallelTrainer
from agent_assets import tools
import agent_assets.A_hparameters as hp
import argparse
from tensorflow.profiler.experimental import Profile
from datetime import timedelta

ENVIRONMENT = 'BipedalWalkerHardcore-v3'

env_kwargs = [
    dict(
    ),
] * 32
env_names = [ENVIRONMENT]*len(env_kwargs)

hp.CLASSIC = True

model_f = am.classic_dense_vmpo

hp.Actor_activation = 'tanh'

evaluate_f = tools.evaluate_common

parser = argparse.ArgumentParser()
parser.add_argument('--step', dest='total_steps',default=100000, type=int)
parser.add_argument('-n','--logname', dest='log_name',default=None)
parser.add_argument('-pf', dest='profile',action='store_true',default=False)
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)
parser.add_argument('-l','--load', dest='load', default=None)
args = parser.parse_args()

total_steps = int(args.total_steps)

hp.Algorithm = 'V-MPO'

hp.Batch_size = 2
hp.Buf.N = 8
hp.k_train_step = 2
hp.Q_discount = 0.95
hp.Target_update = 20

hp.Model_save = 20000
hp.histogram = 1000
hp.log_per_steps = 99
hp.log_grad_per_steps = 9
hp.log_actions = 99

hp.lr['common'].halt_steps = 0
hp.lr['common'].start = 1e-4
hp.lr['common'].end = 1e-4
hp.lr['common'].nsteps = 2e4
hp.lr['common'].epsilon = 1e-5
hp.lr['common'].grad_clip = None

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

p_trainer = ParallelTrainer(
    model_f=model_f,
    m_dir = args.load,
    log_name=args.log_name,
    mixed_float = args.mixed_float,
    env_names=env_names,
    env_kwargs=env_kwargs,
)

if args.profile:
    p_trainer.train_n_steps(20)
    
    with Profile(f'logs/{args.log_name}'):
        p_trainer.train_n_steps(3)
    
    p_trainer.train_n_steps(total_steps-23, evaluate_f)


else :
    p_trainer.train_n_steps(total_steps, evaluate_f)

p_trainer.save_and_evaluate(evaluate_f)
d = timedelta(seconds=time.time() - st)
print(f'{total_steps}steps took {d}')

