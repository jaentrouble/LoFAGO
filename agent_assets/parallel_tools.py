import ray
import numpy as np
from . import A_hparameters as hp
from . import tools
from .replaybuffer import ReplayBufferMulti
from .Agent import Player
import gym
from tqdm import tqdm
import tensorflow as tf

class MultiEnvs():
    """MultiEnvs
    
    Gets batches of actions and returns batches of results
    All environments are expected to have the same observation spaces and
    action spaces

    If render is called, it will call the first environment's
    render method.

    Available methods: step, reset, render(of only the first env)
    """
    def __init__(self, envs:list):
        """
        arguments
        ---------
        envs : list of gym envs
        """
        if not ray.is_initialized():
            ray.init()
        self._envs = envs
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self._r_envs = [RemoteEnv.remote(e) for e in envs]

    def step(self, action):
        """step
        The first axis of action is the batch axis
        i.e [a1, a2, a3, a4, ...] where a_k = action of env_k

        Returns are aggregated in a numpy array, except info.
        Informations will be given in a list.
        i.e. rewards=[r1, r2, ...] where r_k = reward of env_k

        Observations are aggregated in a way tensorflow does.
        i.e. observations = {
            name1 : [o1, o2, o3, ...]
        }

        """
        # [ref(o1,r1,d1,i1),ref(o2,r2,d2,i2),...]
        r_refs = [r_e.step.remote(a) for r_e, a in zip(self._r_envs, action)]
        obss, rews, dons, infs = zip(*ray.get(r_refs))
        o_aggr = {}
        for name in self.observation_space:
            o_aggr[name] = np.stack([o[name] for o in obss])
        r_aggr = np.stack(rews)
        d_aggr = np.stack(dons)
        return o_aggr, r_aggr, d_aggr, infs

    def reset(self, indices:list):
        """reset
        The environments of the given indices will be reset.
        Even when calling for a single environment,
        'indices' argument should be given in a list.

        Observations will be aggregated as 'step' method
        """
        o_refs = []
        for i in indices:
            o_refs.append(self._r_envs[i].reset.remote())
        obss = ray.get(o_refs)
        o_aggr = {}
        for name in self.observation_space:
            o_aggr[name] = np.stack([o[name] for o in obss])
        return o_aggr
    
    def reset_and_swap(self, indices, observation):
        """reset_and_swap
        The environments of the given indices will be reset.
        Then, the corresponding original observations are swapped to
        the new observations returned from the reset method.

        'indices' argument should be given in a list
        """
        o_aggr = self.reset(indices)
        for name, arr in observation.items():
            arr[indices] = o_aggr[name]
        return observation

    def step_auto_reset(self, action):
        """step_auto_reset
        Automatically resets done environments and swap observations
        """
        o, r, d, i = self.step(action)
        if np.any(d):
            d_indices = np.nonzero(d)[0]
            o = self.reset_and_swap(d_indices, o)
        return o, r, d, i

    def reset_all(self):
        """reset_all
        Reset all environments
        """
        return self.reset(list(range(len(self._envs))))
    
    def render(self, *args, **kwargs):
        """render
        Only calls the render method of the first environment.
        """
        render_ref = self._r_envs[0].render.remote(*args, **kwargs)
        return ray.get(render_ref)

    def get_wrapped_env(self, env_index):
        """get_wrapped_env
        Returns an environment of the index wrapped to use as a
        local environment.
        For evaluation purpose
        """
        return SingleEnvWrapper(self._r_envs[env_index],
                                self.observation_space, self.action_space)

class SingleEnvWrapper():
    """SingleEnvWrapper
    Wraps a single remote env to use as a normal local environment.
    For use in evaluation.
    """
    def __init__(self, r_env, observation_space, action_space):
        self._r_env = r_env
        self.observation_space = observation_space
        self.action_space = action_space
    
    def step(self, action):
        return ray.get(self._r_env.step.remote(action))

    def reset(self):
        return ray.get(self._r_env.reset.remote())

    def render(self, *args, **kwargs):
        return ray.get(self._r_env.render.remote(*args, **kwargs))

    def close(self):
        return ray.get(self._r_env.close.remote())


@ray.remote
class RemoteEnv():
    """RemoteEnv
    Wrapper of an environment to execute as a remote actor
    
    Available methods: step, reset, render
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

class ParallelTrainer():
    def __init__(self, model_f, m_dir, log_name, mixed_float,
                       env_names, env_kwargs):
        """
        All environments are expected to have the same action space and
        observation space

        arguments
        ---------
        model_f : function to make model
        m_dir : directory of savedfiles(if loading)
        log_name : name of this test
        mixed_float : whether to use mixed_precision
        env_names: list of str
            list of environment names. This need not be all same, but
            each names should match with corresponding env_kwargs
        env_kwargs: list of dict
            list of environment kwargs.
        """
        envs = []
        for e_n, e_k in zip(env_names, env_kwargs):
            env = gym.make(e_n, **e_k)
            if hp.CLASSIC:
                env = tools.EnvWrapper(env)
            envs.append(env)
        self._player = Player(
            observation_space=env.observation_space,
            action_space=env.action_space,
            model_f=model_f,
            m_dir=m_dir,
            log_name=log_name,
            mixed_float=mixed_float,
        )
        self._env_n = len(envs)
        self._mult_envs = MultiEnvs(envs)
        self._buf = ReplayBufferMulti(env.observation_space, env.action_space)
        self._reset_buffer = True
        self._need_to_eval = False
        self._cum_rewards = np.zeros(self._env_n)
        self._per_round_steps = np.zeros(self._env_n)
        self._rounds = 0
        self._act_steps = 0
        self._last_obs = None
    
    def train_n_steps(self, steps, eval_f=None):
        if eval_f is None:
            print('\n')
            print("*"*80)
            print("Warning: No eval function so not evaluating")
            print("*"*80)
            print('\n')
            evaluate = False
        else:
            evaluate = True
        step_tqdm = tqdm(total=steps, dynamic_ncols=True, leave=True)
        if self._last_obs is None:
            self._last_obs = self._mult_envs.reset_all()

        while step_tqdm.n < steps:
            if ((step_tqdm.n % hp.Model_save) in range(hp.k_train_step))\
                and evaluate:
                self._need_to_eval = True

            if self._reset_buffer:
                self._buf.reset_all()
                explore_n = hp.Batch_size+hp.Buf.N
                self._reset_buffer = False
            else:
                self._buf.reset_continue()
                explore_n = hp.Batch_size
            for _ in range(explore_n):
                self._act_steps += self._env_n
                actions = self._player.act_batch(self._last_obs)
                new_obs, r, d, _ = self._mult_envs.step_auto_reset(actions)
                self._buf.store_step(self._last_obs, actions, r, d)
                
                if self._act_steps%hp.log_actions in range(self._env_n):
                    # Record the first env's action
                    with self._player.file_writer.as_default():
                        tf.summary.scalar('a0', actions[0][0],self._act_steps)
                        if not hp.CLASSIC:
                            tf.summary.scalar('a1', 
                                    actions[0][1],self._act_steps)

                self._cum_rewards += r
                self._per_round_steps += 1

                for done_i in np.nonzero(d)[0]:
                    self._rounds += 1
                    step_tqdm.set_postfix({
                        'Round': self._rounds,
                        'Steps': self._per_round_steps[done_i],
                        'Reward': self._cum_rewards[done_i],
                    })
                    with self._player.file_writer.as_default():
                        tf.summary.scalar('Reward',
                                          self._cum_rewards[done_i],
                                          self._rounds)
                        tf.summary.scalar('Reward_step',
                                          self._cum_rewards[done_i],
                                          self._player.total_steps)
                        tf.summary.scalar('Steps_per_round',
                                          self._per_round_steps[done_i],
                                          self._rounds)

                    self._cum_rewards[done_i] = 0
                    self._per_round_steps[done_i] = 0

                    if self._need_to_eval:
                        self._player.save_model()
                        eval_env = self._mult_envs.get_wrapped_env(done_i)
                        score = eval_f(self._player, eval_env, 'mp4')
                        print(f'eval_score:{score}')
                        self._need_to_eval = False
                        
                        new_obs = self._mult_envs.reset_and_swap([done_i],
                                                                 new_obs)
                self._last_obs = new_obs
            for _ in range(hp.k_train_step):
                self._reset_buffer = self._player.step(self._buf) \
                                     or self._reset_buffer
            step_tqdm.update(n=hp.k_train_step)
        step_tqdm.close()

    def save_and_evaluate(self, eval_f):
        """
        This evaluates with the first environment.
        Note that it will reset the environment.
        """
        self._player.save_model()
        eval_env = self._mult_envs.get_wrapped_env(0)
        score = eval_f(self._player, eval_env, 'mp4')
        print(f'eval_score:{score}')