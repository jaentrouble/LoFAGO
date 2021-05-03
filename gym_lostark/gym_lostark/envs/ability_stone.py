import gym
from gym import spaces
import numpy as np

MAX_PROB = 0.75
MIN_PROB = 0.25
# If primary choice is not available
SECONDARY = [1, 0, 0]
TERTIARY = [2, 2, 1]

class AbilityStone(gym.Env):
    """AbilityStone
    LostArk Ability stone simulator

    Observation Space
    [
        buff_0_left_chance,
        buff_1_left_chance,
        debuff_left_chance,

        buff_0_target_left,
        buff_1_target_left,
        debuff_target_left,

        success_probability
    ]

    Action space
        0 : buff_0
        1 : buff_1
        2 : debuff

        If buff_0 is all filled, 0 becomes 1
        If buff_1 is all filled, 1 becomes 0
        If debuff is all filled, 2 becomes 0

        If only one is left, any actions wil choose the left one

    If target is met (buff_0, buff_1 have reached target success
                      and debuff has failed enough), reward 1 is given.
    From then, if the stone gets any better (buffs success or debuff fails)
        reward increases by 1.
    
    For example, if the target is 7/6/2 and all debuff chances are used,
    7/5/2 -> 7/6/2 : Reward 1
    7/6/2 -> 7/6/2 : Reward 1

    9/5/2 -> 9/5/2 : Reward 0
    9/5/2 -> 9/6/2 : Reward 3

    If target cannot be reached anyhow, the game ends with reward -1

    Game ends when no chances are left
    """
    def __init__(self, **kwargs):
        """
        kwargs
        ------
        chances : int
            Initial number of chances available to try (ex. Legend = 9)
            Defaults to 10
        """
        kwargs.setdefault('chances', 10)
        self._default_chance = np.int32(kwargs['chances'])
        self._target_result = np.array([10,10,0])

        self._stone = np.ones(3)*self._default_chance
        self._successes = np.zeros(3)
        self._fails = np.zeros(3)

        self._rng = np.random.default_rng()

        self._prob = MAX_PROB

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([0.0]*6 + [MIN_PROB]),
            high=np.array([self._default_chance]*6+[MAX_PROB]),
            dtype=np.float32
        )

        self._done = False
        self._initialized = False


    def step(self, action):
        assert self._initialized, 'Reset first before starting env'
        assert self.action_space.contains(action), 'Invalid action!'
        assert np.all(self._stone>=0), 'Negative available chances, check code'

        reward = 0

        if self._done:
            print ('The game is already done. ',
                    'Continuing may cause unexpected behaviors')
            return self._get_observation(), reward, self._done, None

        
        if self._stone[action] == 0:
            if self._stone[SECONDARY[action]] > 0:
                self._carve_stone(SECONDARY[action])
            else:
                if self._stone[TERTIARY[action]]>0:
                    self._carve_stone(TERTIARY[action])
                else:
                    raise ValueError('No available chances')
        else:
            self._carve_stone(action)

        if (self._target_result[0]<=self._successes[0] and
            self._target_result[1]<=self._successes[1] and
            self._target_result[2]>=self._stone[2]+self._successes[2]):
            reward += 1
            reward += self._successes[0]-self._target_result[0]
            reward += self._successes[1]-self._target_result[1]
            reward += self._target_result[2] - self._stone[2]+self._successes[2]

        elif (self._target_result[0]>self._stone[0]+self._successes[0] or
              self._target_result[1]>self._stone[1]+self._successes[1] or
              self._target_result[2]<self._successes[2]):
              self._done = True
              reward = -1
        
        if np.all(self._stone == 0):
            self._done = True

        return self._get_observation(), reward, self._done, self._successes
        
    def _carve_stone(self, action):
        if self._rng.random()<self._prob:
            self._stone[action] -= 1
            self._successes[action] += 1
            self._prob_success()
        else:
            self._stone[action] -= 1
            self._fails[action] += 1
            self._prob_fail()

    def _prob_fail(self):
        self._prob = min(MAX_PROB, self._prob+0.1)

    def _prob_success(self):
        self._prob = max(MIN_PROB, self._prob-0.1)

    def _get_observation(self):
        target_left = np.maximum(
            self._target_result - self._successes,0
        )
        return np.float32(np.concatenate(
            (self._stone, target_left, [self._prob])
        ))

    
    def reset(self, target_result):
        """reset
        
        Parameters
        ----------
        target_result : [buff_0, buff_1, debuff]
            buff_0 & buff_1 : Minimum successes (Success is good)
            debuff : Maximum successes (Success is bad)
        """
        self._initialized = True
        self._prob = MAX_PROB
        self._stone = np.ones(3)*self._default_chance
        self._target_result = np.array(target_result)
        self._successes = np.zeros(3)
        self._fails = np.zeros(3)

        self._done = False

        return self._get_observation()


    def render(self):
        pass

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed=seed)

    def close(self):
        pass