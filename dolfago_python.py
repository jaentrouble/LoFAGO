import gym, gym_lostark
import numpy as np
import tqdm
import random

TRY_N = 100000
TARGET = (0,0,10)

prob_list = [0.25,0.35,0.45,0.55,0.65,0.75]

def prob_to_idx(prob):
    return int((prob-0.25)/0.1)

EsMap = np.zeros((100, len(prob_list)))
EsMap_filled = np.zeros_like(EsMap, dtype=np.bool)
EaMap = np.zeros((100, 100, len(prob_list)))
EaMap_filled = np.zeros_like(EaMap, dtype=np.bool)
Ea1Map = np.zeros((100, 100, 100, len(prob_list)))
Ea1Map_filled = np.zeros_like(Ea1Map, dtype=np.bool)


def succ_prob(p_idx):
    return min(5, p_idx+1)

def fail_prob(p_idx):
    return max(0, p_idx-1)

def calculate_E(p_idx, succ, fail, same):
    prob=prob_list[p_idx]
    v = prob*succ + (1-prob)*fail
    if same:
        v += prob
    return v

def calculate_Es(n, p_idx):
    return calculate_E(
        p_idx,
        getEs(n, succ_prob(p_idx)),
        getEs(n, fail_prob(p_idx)),
        True
    )

def calculate_Ea(na, nb, p_idx, same):
    return calculate_E(
        p_idx,
        getEa(na, nb, succ_prob(p_idx)),
        getEa(na, nb, fail_prob(p_idx)),
        same
    )

def calculate_Ea1(na1, na2, nb, p_idx, same):
    return calculate_E(
        p_idx,
        getEa1(na1, na2, nb, succ_prob(p_idx)),
        getEa1(na1, na2, nb, fail_prob(p_idx)),
        same
    )

def getEs(n, p_idx):
    if EsMap_filled[n,p_idx]:
        pass
    elif n==0:
        EsMap[n,p_idx] = 0
    else:
        EsMap[n,p_idx] = calculate_Es(n-1, p_idx)
    EsMap_filled[n,p_idx] = True

    return EsMap[n,p_idx]

def getEa(na, nb, p_idx):
    if EaMap_filled[na,nb,p_idx]:
        pass
    elif na==0:
        EaMap[na,nb,p_idx] = 0
    elif nb==0:
        EaMap[na,nb,p_idx] = getEs(na, p_idx)
    else:
        EaA = calculate_Ea(na-1, nb, p_idx, True)
        EaB = calculate_Ea(na, nb-1, p_idx, False)
        EaMap[na,nb,p_idx] = max(EaA, EaB)
    EaMap_filled[na,nb,p_idx] = True

    return EaMap[na,nb,p_idx]

def getEa1(na1, na2, nb, p_idx):
    if Ea1Map_filled[na1,na2,nb,p_idx]:
        pass
    elif na1==0:
        Ea1Map[na1,na2,nb,p_idx] = 0
    elif (na2==0) and (nb==0):
        Ea1Map[na1,na2,nb,p_idx] = getEs(na1, p_idx)
    elif na2==0:
        Ea1Map[na1,na2,nb,p_idx] = getEa(na1, nb, p_idx)
    elif nb==0:
        Ea1Map[na1,na2,nb,p_idx] = getEa(na1, na2, p_idx)
    else:
        Ea1A1 = calculate_Ea1(na1-1, na2, nb, p_idx, True)
        Ea1A2 = calculate_Ea1(na1, na2-1, nb, p_idx, False)
        Ea1Map[na1,na2,nb,p_idx] = max(Ea1A1, Ea1A2)
    Ea1Map_filled[na1,na2,nb,p_idx] = True
    
    return Ea1Map[na1,na2,nb,p_idx]

def get_p_table(na1, na2, nb, p_idx):
    na = na1+na2
    n = na + nb
    Es = 0
    EaA = 0
    EbA = 0
    EaB = 0
    EbB = 0
    Ea1A1 = 0
    Ea1A2 = 0
    Ea2A1 = 0
    Ea2A2 = 0
    Es = getEs(n, p_idx)
    if na>0:
        EaA = calculate_Ea(na-1, nb, p_idx, True)
        EbA = Es - EaA
    if nb>0:
        EaB = calculate_Ea(na, nb-1, p_idx, False)
        EbB = Es - EaB
    if na1>0:
        Ea1A1 = calculate_Ea1(na1-1, na2, nb, p_idx, True)
        Ea2A1 = EaA - Ea1A1
    if na2>0:
        Ea1A2 = calculate_Ea1(na1, na2-1, nb, p_idx, False)
        Ea2A2 = EaA - Ea1A2
    return [
        [Ea1A1, Ea2A1, EbA],
        [Ea1A2, Ea2A2, EbA],
        [EaB, EaB, EbB],
        max(EaA, EaB)
    ]
env = gym.make('AbilityStone-v0')
results = []
for _ in tqdm.trange(TRY_N):
    o = env.reset(TARGET)
    done = False
    while not done:
        na1, na2, nb, _,_,_, prob = o
        na1 = int(na1)
        na2 = int(na2)
        nb = int(nb)
        p_idx = prob_to_idx(prob)
        p_table = get_p_table(na1, na2, nb, p_idx)
        if (p_table[0][2]==0 and
            p_table[2][0]==0 and
            p_table[2][2]>0):
            action = 2
        elif p_table[0][2] > p_table[2][2]:
            action = 2
        elif (p_table[0][0] > p_table[1][0] and
              p_table[0][1] < p_table[1][1]):
            action = 0
        elif (p_table[0][0] < p_table[1][0] and
              p_table[0][1] > p_table[1][1]):
            action = 1
        else:
            action = random.randint(0,1)
        o, r, done, i = env.step(action)
    results.append(i)
np.savetxt(f'savefiles/dolphago/eval_{TARGET}_{TRY_N}.csv',
            np.array(results),delimiter=',')
