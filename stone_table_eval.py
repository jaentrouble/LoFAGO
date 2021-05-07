import gym, gym_lostark
from multiprocessing import Process, Queue
import tqdm
import numpy as np


def prob_to_idx(prob):
    return round((prob-0.25)/0.1)

def p_fail(p_idx):
    return min(5,p_idx+1)

def p_succ(p_idx):
    return max(0,p_idx-1)

def test_proc(Q, TRY_N,target, table):
    env = gym.make('AbilityStone-v0')
    results = []
    for _ in range(TRY_N):
        o = env.reset(target)
        done = False
        while not done:
            c1, c2, c3, t1, t2, t3, prob = o
            c1 = int(c1)
            c2 = int(c2)
            c3 = int(c3)
            t1 = int(t1)
            t2 = int(t2)
            t3 = int(t3)
            p_idx = prob_to_idx(prob)
            action = np.argmax(table[c1,c2,c3,t1,t2,t3,p_idx])
            o, r, done, i = env.step(action)
        results.append(i)
    Q.put(results)


if __name__ == '__main__':

    MAX_PROCS = 12
    MULTIPLIER = 100
    TRY_N = 10000
    table = np.load('exp_table.npz')['table']
    result_Q = Queue()
    target_tqdm = tqdm.tqdm(total=9)
    
    for t1 in range(7,8):
        for t2 in range(5,t1+1):
            for t3 in range(1,4):
                target = [t1,t2,t3]
                target_tqdm.set_description(str(target))
                proc_tqdm = tqdm.tqdm(total=MULTIPLIER, leave=False)
                for _ in range(min(MULTIPLIER,MAX_PROCS)):
                    Process(target=test_proc, args=(result_Q, TRY_N, target,table)).start()
                results_list = []
                done_proc = 0
                while done_proc<MULTIPLIER:
                    if not result_Q.empty():
                        results_list.extend(result_Q.get())
                        done_proc+= 1
                        proc_tqdm.update()
                        left_over = MULTIPLIER-done_proc-MAX_PROCS
                        if left_over>=0:
                            Process(target=test_proc, args=(result_Q, TRY_N, target, table)).start()
                proc_tqdm.close()
                target_tqdm.update()
                np.savetxt(f'savefiles/stone_table/eval_{target}_{TRY_N}x{MULTIPLIER}.csv',
                            np.array(results_list),delimiter=',')
