import numpy as np
from bingo_table_maker import *
table = np.load('bingo_tables/bingo_table.npz')['table']

def pretty_print_table(print_table):
    new_table = np.where(print_table, '⬤','◯')
    print(new_table)

def all_action_max_weak(target_table, step):
    max_weak_table = np.zeros_like(target_table, dtype=np.int)
    for x in range(5):
        for y in range(5):
            next_table = bomb_explode(target_table,(x,y))
            next_index = tuple(np.append(next_table.reshape(-1),(step+1)%3))
            _, _, next_weak, _, _ = table[next_index].astype(np.int)
            max_weak_table[x,y] = next_weak
    return max_weak_table


initial_table = np.zeros((5,5),dtype=np.bool)
initial_table[2,4] = True
initial_table[4,3] = True
for i in range(19):
    pretty_print_table(initial_table)
    table_index = tuple(np.append(initial_table.reshape(-1),i%3))
    x, y, max_weak, bingo_p, skull_p = table[table_index].astype(np.int)
    # print('x:',x,' y:',y,' weak:',max_weak,' b_p:',bingo_p,' s_p:',skull_p)
    print(all_action_max_weak(initial_table,i))
    initial_table = bomb_explode(initial_table, (x,y))
