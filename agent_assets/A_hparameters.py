# Per-env batch size, not global batch size
Batch_size = 192
Target_update = 100
Q_discount = 0.99

k_train_step = 8

Actor_activation = 'tanh'

available_algorithms=(
    'V-MPO',
    'PPO',
    'A2C',
)
Algorithm = 'V-MPO'

class Lr():
    def __init__(self):
        self.start = None
        self.end = None
        self.halt_steps = None
        self.nsteps = None
        self.epsilon = None
        self.grad_clip = None

lr = {
    'common' : Lr(),
    'actor' : Lr(),
    'critic' : Lr(),
    'encoder' : Lr(),
}
lr['common'].halt_steps = 0
lr['common'].start = 1e-4
lr['common'].end = 1e-4
lr['common'].nsteps = 2000000
lr['common'].epsilon = 1e-2
lr['common'].grad_clip = 1.0

lr['actor'].halt_steps = 0
lr['actor'].start = 0.001
lr['actor'].end = 0.00005
lr['actor'].nsteps = 2000000
lr['actor'].epsilon = 1e-2
lr['actor'].grad_clip = 1.0

lr['critic'].halt_steps = 0
lr['critic'].start = 0.001
lr['critic'].end = 0.00005
lr['critic'].nsteps = 2000000
lr['critic'].epsilon = 1e-2
lr['critic'].grad_clip = 1.0

lr['encoder'].halt_steps = 0
lr['encoder'].start = 1e-4
lr['encoder'].end = 1e-5
lr['encoder'].nsteps = 1e6
lr['encoder'].epsilon = 1e-2
lr['encoder'].grad_clip = 1.0

lr['forward'] = lr['encoder']
lr['inverse'] = lr['encoder']

CLASSIC = False

IQN_ENABLE = True
IQN_SUPPORT = 64
IQN_COS_EMBED = 64

ICM_ENABLE = True
ICM_intrinsic = 1.0
ICM_loss_forward_weight = 0.2

VMPO_eps_eta = 1e-1
VMPO_eps_alpha_mu = 1e-2
VMPO_eps_alpha_sig = 1e-5
VMPO_eta_min = 1e-5
VMPO_alpha_min = 1e-5

PPO_eps_clip = 0.2

A2C_sig_max = 50
A2C_sig_min = 1e-2

class _Buf():
    def __init__(self):
        self.N = 64

Buf = _Buf()

Model_save = 1000

histogram = 1000
log_per_steps = 100
log_grad_per_steps = 1

log_actions = 10