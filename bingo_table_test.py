import numpy as np
from bingo_table_maker import *
table = np.load('bingo_tables/1143_table.npz')['table']

def pretty_print_table(table):
    new_table = np.where(table, '⬤','◯')
    print(new_table)


initial_table = np.zeros((5,5),dtype=np.bool)
initial_table[1,1] = True
initial_table[4,3] = True
for i in range(19):
    pretty_print_table(initial_table)
    table_index = tuple(np.append(initial_table.reshape(-1),i%3))
    x, y, max_weak, bingo_p, skull_p = table[table_index].astype(np.int)
    print('x:',x,' y:',y,' weak:',max_weak,' b_p:',bingo_p,' s_p:',skull_p)
    initial_table = bomb_explode(initial_table, (x,y))