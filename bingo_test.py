import numpy as np
from bingo_table_maker import count_bingo, bomb_explode

table = np.load('bingo_tables/merged_comp.npz')['table']
count = 0
for i in range(25):
    for j in range(25):
        init_board = np.zeros((5,5),dtype=bool)
        init_board[i//5,i%5] = True
        init_board[j//5,j%5] = True
        for s in [3,4,5,0,1,2]:
            idx = np.append(init_board.reshape(-1).astype(int),s)
            idx = tuple(idx)
            result = table[idx]
            next_x, next_y = int(result[0]), int(result[1])

            init_board = bomb_explode(init_board, (next_x,next_y))
        if count_bingo(init_board)!=1:
            print([i//5,i%5,j//5,j%5])
            count+=1
print(count)