import numpy as np

p_list = [0.25,0.35,0.45,0.55,0.65,0.75]
FAIL = 0
SUCCESS = 1
EXCESS = 2
NS = 3
REWARDS = [0,1,0,0] # Fail, Success, Excess, NS
q_table = np.zeros((11,11,11,11,11,11,6,3))
q_filled = np.zeros((11,11,11,11,11,11,6),dtype=bool)

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
            if t2<=0 and c3<=t3:
                if t1 == 1:
                    return SUCCESS
                elif t1<=0:
                    return EXCESS
                else: return NS
            else:
                return NS
        else:
            if c1-1<t1:
                return FAIL
            else:
                return NS
    elif idx == 2:
        if successed:
            if t1<=0 and c3 <= t3:
                if t2==1:
                    return SUCCESS
                elif t2<=0:
                    return EXCESS
                else: return NS
            else:
                return NS
        else:
            if c2-1<t2:
                return FAIL
            else:
                return NS
    elif idx == 3:
        if successed:
            if t3-1 < 0:
                return FAIL
            else:
                return NS
        else:
            if t1<=0 and t2<=0:
                if c3 == t3+1:
                    return SUCCESS
                elif c3<=t3:
                    return EXCESS
                else: return NS
            else:
                return NS
    raise ValueError('unhandled case!!')


def get_q(c1, c2, c3, t1, t2, t3, p_idx):
    t1 = max(t1,0)
    t2 = max(t2,0)
    t3 = max(t3,0)
    if c1==0 and c2==0 and c3==0:
        return [0,0,0]
    if q_filled[c1,c2,c3,t1,t2,t3,p_idx]:
        return q_table[c1,c2,c3,t1,t2,t3,p_idx]
        
    if c1 > 0:
        q1_s = check_reward(c1,c2,c3,t1,t2,t3,True,1)
        q1_sr = REWARDS[q1_s]
        q1_f = check_reward(c1,c2,c3,t1,t2,t3,False,1)
        q1_fr = REWARDS[q1_f]
        q1= p_list[p_idx]*((q1_s!=FAIL)*np.max(get_q(max(c1-1,0),c2,c3,max(t1-1,0),t2,t3,prob_success(p_idx))) + \
                            q1_sr) +\
            (1-p_list[p_idx])*((q1_f!=FAIL)*np.max(get_q(max(c1-1,0),c2,c3,t1,t2,t3,prob_fail(p_idx))) +\
                               q1_fr)
    else:
        q1 = 0
    if c2 > 0:
        q2_s = check_reward(c1,c2,c3,t1,t2,t3,True,2)
        q2_sr = REWARDS[q2_s]
        q2_f = check_reward(c1,c2,c3,t1,t2,t3,False,2)
        q2_fr = REWARDS[q2_f]
        q2= p_list[p_idx]*((q2_s!=FAIL)*np.max(get_q(c1,max(c2-1,0),c3,t1,max(t2-1,0),t3,prob_success(p_idx))) + \
                            q2_sr) +\
            (1-p_list[p_idx])*((q2_f!=FAIL)*np.max(get_q(c1,max(c2-1,0),c3,t1,t2,t3,prob_fail(p_idx))) +\
                               q2_fr)
    else:
        q2 = 0
    if c3 > 0:
        q3_s = check_reward(c1,c2,c3,t1,t2,t3,True,3)
        q3_sr = REWARDS[q3_s]
        q3_f = check_reward(c1,c2,c3,t1,t2,t3,False,3)
        q3_fr = REWARDS[q3_f]
        q3= p_list[p_idx]*((q3_s!=FAIL)*np.max(get_q(c1,c2,max(c3-1,0),t1,t2,max(t3-1,0),prob_success(p_idx))) + \
                           q3_sr) +\
            (1-p_list[p_idx])*((q3_f!=FAIL)*np.max(get_q(c1,c2,max(c3-1,0),t1,t2,t3,prob_fail(p_idx))) +\
                               q3_fr)
    else:
        q3 = 0

    q_filled[c1,c2,c3,t1,t2,t3,p_idx] = True
    q_table[c1,c2,c3,t1,t2,t3,p_idx] = np.array([q1,q2,q3])

    return np.array([q1,q2,q3])

if __name__ == '__main__':
    
    from tqdm import trange
    for t1 in trange(10,leave=False):
        for t2 in trange(10,leave=False):
            for t3 in trange(10,leave=False):
                _ = get_q(10,10,10,t1+1,t2+1,t3+1,5)

    np.savez('exp_table.npz',table=q_table)
