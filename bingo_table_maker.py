import numpy as np
import tqdm

# x, y, Max 무력, Bingo 점수, 해골 위치 점수
# Bingo 점수, 해골 점수: Lower is better
q_table = np.zeros([2]*25+[3,5])
q_filled = np.zeros([2]*25+[3], dtype=np.bool)
MAX_STEPS = 18
EMPTY_BOARD = np.zeros((5,5), dtype=np.bool)
B_POINT = np.array([1,4,3,2,0])
B_N_PENALTY = 5
KUKU = np.array([1,1])

def skull_point(table):
    point = 0
    k_x, k_y = KUKU
    for x, y in zip(*np.nonzero(table)):
        if x==k_x and y==k_y:
            point += 4
        elif abs(x-k_x)<=1 and abs(y-k_y)<=1:
            point += 3
        elif abs(x-k_x)<=2 and abs(y-k_y)<=2:
            point += 2
        elif abs(x-k_x)<=3 and abs(y-k_y)<=3:
            point += 1
    return point

def bingo_point(before_bingo_idx, next_bingo_idx):
    """
    before_bingo/next_bingo: bingo indices

    """
    point = 0
    if len(before_bingo_idx)<5:
        new_bingos = []
        left_bingos = list(range(5))
        for bb in before_bingo_idx:
            left_bingos.remove(bb)
        for nb in next_bingo_idx:
            if not(nb in before_bingo_idx):
                new_bingos.append(nb)
        left_bingo_points = [B_POINT[lb] for lb in left_bingos]
        left_bingo_min = np.min(left_bingo_points)
        new_bingo_points = [B_POINT[nwb] for nwb in new_bingos]
        for nwbp in new_bingo_points:
            point += nwbp - left_bingo_min
    return point

def count_bingo(table):
    x, y = check_bingo(table)
    return len(x) + len(y)

def check_bingo(table):
    x_bingo = np.logical_and.reduce(table, axis=1)
    y_bingo = np.logical_and.reduce(table, axis=0)
    return np.nonzero(x_bingo)[0], np.nonzero(y_bingo)[0]

def bomb_explode(table, bomb_pos):
    """
    bomb_pos : (x, y)
    """
    next_table = table.copy()
    x, y = bomb_pos
    x_bingo, y_bingo = check_bingo(next_table)
    if not (y in y_bingo):
        if x-1>=0 and not((x-1) in x_bingo):
            next_table[x-1,y] = not next_table[x-1,y]
        if x+1<5 and not ((x+1) in x_bingo):
            next_table[x+1,y] = not next_table[x+1,y]
    if not (x in x_bingo):
        if y-1>=0 and not((y-1) in y_bingo):
            next_table[x,y-1] = not next_table[x,y-1]
        if y+1<5 and not ((y+1) in y_bingo):
            next_table[x,y+1] = not next_table[x,y+1]
    if not (x in x_bingo) and not (y in y_bingo):
        next_table[x,y] = not next_table[x,y]
    return next_table


def fill_table(initial_table):
    state_stack = [(initial_table,0)]
    stack_tqdm = tqdm.tqdm(total=1)
    max_stack = 1
    loop_n = 0
    while len(state_stack)>0:
        current_table, step = state_stack[-1]
        current_index = np.concatenate((current_table.reshape(-1),[step%3]))
        current_index = tuple(current_index)
        possible_choices = []
        need_to_fill = False
        for action_x in range(5):
            for action_y in range(5):
                next_table = bomb_explode(current_table, (action_x,action_y))
                before_bingo = count_bingo(current_table)
                before_bingo_x, before_bingo_y = check_bingo(current_table)
                next_bingo = count_bingo(next_table)
                next_bingo_x, next_bingo_y = check_bingo(next_table)
                assert next_bingo>=before_bingo, 'Bingo number decreased!'
                
                new_bingo = next_bingo - before_bingo
                next_index = np.concatenate((next_table.reshape(-1),
                                             [(step+1)%3]))
                next_index = tuple(next_index)
                if (step%3 == 2) and new_bingo == 0:
                    # Game over
                    possible_choices.append([
                        action_x,
                        action_y,
                        0,
                        0,
                        0
                    ])
                elif q_filled[next_index]:
                    if (step%3==2) and new_bingo>0:
                        # 무력 성공
                        possible_choices.append([
                            action_x,
                            action_y,
                            q_table[next_index][2]+1,
                            q_table[next_index][3]\
                                +bingo_point(before_bingo_x, next_bingo_x)\
                                +bingo_point(before_bingo_y, next_bingo_y)\
                                +B_N_PENALTY*(new_bingo-1),
                            q_table[next_index][4]+skull_point(next_table)
                        ])
                    else:
                        # No need to weak kuku,
                        # Unnecessary bingo
                        possible_choices.append([
                            action_x,
                            action_y,
                            q_table[next_index][2],
                            q_table[next_index][3]\
                                +bingo_point(before_bingo_x, next_bingo_x)\
                                +bingo_point(before_bingo_y, next_bingo_y)\
                                +B_N_PENALTY*new_bingo,
                            q_table[next_index][4]+skull_point(next_table)
                        ])
                else:
                    state_stack.append((next_table, step+1))
                    need_to_fill = True
        if not need_to_fill:
            # All checked
            best_choices = []
            max_weak = 0
            for pc in possible_choices:
                if max_weak<pc[2]:
                    best_choices = []
                    best_choices.append(pc)
                elif max_weak==pc[2]:
                    best_choices.append(pc)
            min_bingo_p = best_choices[0][3]
            best_choices_2 = []
            for bc in best_choices:
                if min_bingo_p>bc[3]:
                    best_choices_2=[]
                    best_choices_2.append(bc)
                elif min_bingo_p == bc[3]:
                    best_choices_2.append(bc)
            min_skull_p = best_choices_2[0][4]
            best_choices_3 = []
            for bc2 in best_choices_2:
                if min_skull_p>bc2[4]:
                    best_choices_3=[]
                    best_choices_3.append(bc2)
                elif min_skull_p==bc2[4]:
                    best_choices_3.append(bc2)

            q_table[current_index] = best_choices_3[0]
            q_filled[current_index] = True
            state_stack.pop()

        if len(state_stack)>max_stack:
            max_stack = len(state_stack)
            stack_tqdm.total = max_stack
        stack_tqdm.n = len(state_stack)
        loop_n += 1
        stack_tqdm.set_description(f'loop: {loop_n}')
        stack_tqdm.update(n=0)
    stack_tqdm.close()

if __name__ == '__main__':
    initial_table = np.array([
        [0,0,0,0,0],
        [0,1,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,1,0]
    ])
    fill_table(initial_table)
    np.savez_compressed('1143_table.npz',table=initial_table)