import numpy as np

p_list = [0.25,0.35,0.45,0.55,0.65,0.75]

def prob_fail(p_idx):
    return min(5,p_idx+1)

def prob_success(p_idx):
    return max(0,p_idx-1)

def check_reward(c1,c2,c3,t1,t2,t3, successed, idx):
    """
    What happens if 'idx' 'successed' from 'this state'?
    """
    if idx == 1 :
        if successed:
            if t1-1<=0 and t2==0 and c3<=t3:
                return 1
            else:
                return 0
        else:
            if c1-1<t1:
                return -1
            else:
                return 0
    elif idx == 2:
        if successed:
            if t2-1<=0 and t1==0 and c3 <= t3:
                return 1
            else:
                return 0
        else:
            if c2-1<t2:
                return -1
            else:
                return 0
    elif idx == 3:
        if successed:
            if t3-1 < 0:
                return -1
            else:
                return 0
        else:
            if c3-1<=t3:
                return 1
            else:
                return 0
    raise ValueError('unhandled case!!')

e_table = np.zeros((11,11,11,11,11,11,6))
e_filled = np.zeros_like(e_table,dtype=bool)

def calculate_exp(c1, c2, c3, t1, t2, t3, p_idx):
    t1 = max(t1,0)
    t2 = max(t2,0)
    t3 = max(t3,0)
    if c1==0 and c2==0 and c3==0:
        return 0
    if e_filled[c1,c2,c3,t1,t2,t3,p_idx]:
        return e_table[c1,c2,c3,t1,t2,t3,p_idx]
    e_list = []
    if c1 > 0:
        e1= p_list[p_idx]*(calculate_exp(max(c1-1,0),c2,c3,max(t1-1,0),t2,t3,prob_success(p_idx)) + \
                            check_reward(c1,c2,c3,t1,t2,t3,True,1)) +\
            (1-p_list[p_idx])*(calculate_exp(max(c1-1,0),c2,c3,t1,t2,t3,prob_fail(p_idx)) +\
                                check_reward(c1,c2,c3,t1,t2,t3,False,1))
        e_list.append(e1)
    if c2 > 0:
        e2= p_list[p_idx]*(calculate_exp(c1,max(c2-1,0),c3,t1,max(t2-1,0),t3,prob_success(p_idx)) + \
                            check_reward(c1,c2,c3,t1,t2,t3,True,2)) +\
            (1-p_list[p_idx])*(calculate_exp(c1,max(c2-1,0),c3,t1,t2,t3,prob_fail(p_idx)) +\
                                check_reward(c1,c2,c3,t1,t2,t3,False,2))
        e_list.append(e2)
    if c3 > 0:
        e3= p_list[p_idx]*(calculate_exp(c1,c2,max(c3-1,0),t1,t2,max(t3-1,0),prob_success(p_idx)) + \
                            check_reward(c1,c2,c3,t1,t2,t3,True,3)) +\
            (1-p_list[p_idx])*(calculate_exp(c1,c2,max(c3-1,0),t1,t2,t3,prob_fail(p_idx)) +\
                                check_reward(c1,c2,c3,t1,t2,t3,False,3))
        e_list.append(e3)

    e_max = np.max(e_list)
    e_filled[c1,c2,c3,t1,t2,t3,p_idx] = True
    e_table[c1,c2,c3,t1,t2,t3,p_idx] = e_max

    return e_max

from tqdm import trange
for t1 in trange(10,leave=False):
    for t2 in trange(10,leave=False):
        for t3 in trange(10,leave=False):
            _ = calculate_exp(10,10,10,t1+1,t2+1,t3+1,5)

np.savez('exp_table.npz',table=e_table)
