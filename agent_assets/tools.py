import cv2
from os import path, makedirs
import numpy as np
from .replaybuffer import ReplayBuffer
from . import A_hparameters as hp
from tqdm import tqdm
from .Agent import Player
import tensorflow as tf

def evaluate_unity(player, env, video_type):
    print('Evaluating...')
    done = False
    if player.model_dir is None:
        eval_dir = path.join(player.save_dir,'eval')
        if not path.exists(eval_dir):
            makedirs(eval_dir)
    else:
        eval_dir = player.model_dir
    eye_dir = path.join(eval_dir,f'eval_eye.{video_type}')
    ren_dir = path.join(eval_dir,f'eval_ren.{video_type}')
    score_dir = path.join(eval_dir,'score.txt')

    if 'avi' in video_type :
        fcc = 'DIVX'
    elif 'mp4' in video_type:
        fcc = 'mp4v'
    else:
        raise TypeError('Wrong videotype')
    fourcc = cv2.VideoWriter_fourcc(*fcc)
    # Becareful : cv2 order of image size is (width, height)
    eye_size = env.observation_space['obs'].shape[1::-1]
    # Test image for size
    rend_img = env.render('rgb')
    render_size = (rend_img.shape[1], rend_img.shape[0])
    eye_out = cv2.VideoWriter(eye_dir, fourcc, 10, eye_size)
    ren_out = cv2.VideoWriter(ren_dir, fourcc, 10, render_size)

    o = env.reset()
    score = 0
    loop = 0
    while not done :
        loop += 1
        if not loop % 100:
            print('Eval : {}step passed'.format(loop))
        a = player.act(o)
        o,r,done,i = env.step(a)
        score += r
        #eye recording
        eye_out.write(o['obs'][...,-1:-4:-1])
        ren_out.write(env.render('rgb')[...,::-1])
    ren_out.release()
    eye_out.release()
    with open(score_dir, 'w') as f:
        f.write(str(score))
    print('Eval finished')
    return score

def evaluate_common(player, env, video_type):
    print('Evaluating...')
    done = False
    video_dir = path.join(player.model_dir, 'eval.{}'.format(video_type))
    score_dir = path.join(player.model_dir, 'score.txt')
    if 'avi' in video_type :
        fcc = 'DIVX'
    elif 'mp4' in video_type:
        fcc = 'mp4v'
    else:
        raise TypeError('Wrong videotype')
    fourcc = cv2.VideoWriter_fourcc(*fcc)
    # Becareful : cv2 order of image size is (width, height)
    o = env.reset()
    rend_img = env.render('rgb_array')
    # cv2 expects 90 degrees rotated
    out_shape = (rend_img.shape[1],rend_img.shape[0])
    out = cv2.VideoWriter(video_dir, fourcc, 10, out_shape)
    score = 0
    loop = 0
    while not done :
        loop += 1
        if loop % 100 == 0:
            print('Eval : {}step passed'.format(loop))
        a = player.act(o)
        o,r,done,i = env.step(a)
        score += r
        img = env.render('rgb_array')
        out.write(np.flip(img, axis=-1))
    out.release()
    with open(score_dir, 'w') as f:
        f.write(str(score))
    print('Eval finished')
    env.close()
    return score

class EnvWrapper():
    """Change normal observation into dictionary type
    {
        'obs' : (actual observation)
    }
    """
    def __init__(self, env):
        self.env = env
        from gym import spaces
        self.observation_space = spaces.Dict(
            {'obs' : self.env.observation_space}
        )
        self.action_space = self.env.action_space

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return {'obs':o}, r, d, i

    def reset(self):
        return {'obs':self.env.reset()}

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()



def k_steps(reset_buffer:bool, buf:ReplayBuffer, player:Player, env,
            last_obs, my_tqdm:tqdm, cum_reward:float, rounds:int,act_steps:int,
            per_round_steps:int, render: bool, need_to_eval:bool, k=1,
            eval_f = None):
    """
    1. Fill buffer (if need_to_reset: reset all)
    2. Train player k steps
    """
    evaluated = False
    if reset_buffer:
        buf.reset_all()
        explore_n = hp.Batch_size+hp.Buf.N
    else:
        buf.reset_continue()
        explore_n = hp.Batch_size
    for _ in range(explore_n):
        act_steps += 1
        action = player.act(last_obs)
        new_obs, r, d, _ = env.step(action)
        buf.store_step(last_obs, action, r, d)
        
        if act_steps%hp.log_actions == 0:
            with player.file_writer.as_default():
                tf.summary.scalar('a0', action[0],act_steps)
                if not hp.CLASSIC:
                    tf.summary.scalar('a1', action[1],act_steps)

        cum_reward += r
        per_round_steps += 1

        if render:
            env.render()

        if d:
            if render:
                env.render()
            rounds += 1
            my_tqdm.set_postfix({
                'Round':rounds,
                'Steps': per_round_steps,
                'Reward': cum_reward,
            })
            with player.file_writer.as_default():
                tf.summary.scalar('Reward',cum_reward,rounds)
                tf.summary.scalar('Reward_step',cum_reward,player.total_steps)
                tf.summary.scalar('Steps_per_round',per_round_steps,rounds)

            cum_reward = 0
            per_round_steps = 0
            if need_to_eval:
                player.save_model()
                score = eval_f(player, env, 'mp4')
                print(f'eval_score:{score}')
                evaluated = True
                need_to_eval = False

            
            last_obs = env.reset()
        else:
            last_obs = new_obs
    for _ in range(k):
        reset_buffer = player.step(buf) or reset_buffer
    my_tqdm.update(n=k)

    return (last_obs,
            cum_reward,
            rounds,
            act_steps,
            per_round_steps,
            evaluated,
            reset_buffer)
