import numpy as np


# x, y, Max 무력, Bingo 점수, 해골 위치 점수, bingo 내부 여부
# Bingo 점수, 해골 점수: Lower is better
# 첫번째 무력때만은 다른 상황
q_table = np.zeros([2]*25+[4,6])
q_filled = np.zeros([2]*25+[4], dtype=np.bool)
# Load
q_loaded = np.load('bingo_tables/bingo_table_inanna_neverbingo_backup.npz')
q_loaded_table = q_loaded['table']
q_loaded_filled = q_loaded['filled']
q_table[...,:3,:]=q_loaded_table
q_filled[...,:3]=q_loaded_filled

MAX_STEPS = 18
EMPTY_BOARD = np.zeros((5,5), dtype=np.bool)
B_POINT = np.array([1,4,3,2,0,5])
DIAG_IDX = 5
B_N_PENALTY = 5
KUKU = np.array([1,1])
WEAK_LIMIT = 6

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
    if len(before_bingo_idx)<6:
        new_bingos = []
        left_bingos = list(range(6))
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
    x_bingo_idx = np.nonzero(x_bingo)[0]
    if (table[0,0] and
        table[1,1] and
        table[2,2] and
        table[3,3] and
        table[4,4]):
        x_bingo_idx = np.append(x_bingo_idx, DIAG_IDX)
    y_bingo = np.logical_and.reduce(table, axis=0)
    y_bingo_idx = np.nonzero(y_bingo)[0]
    if (table[0,4] and
        table[1,3] and
        table[2,2] and
        table[3,1] and
        table[4,0]):
        y_bingo_idx = np.append(y_bingo_idx, DIAG_IDX)
    return x_bingo_idx, y_bingo_idx

def is_in_bingo(table, x, y):
    x_bingo_idx, y_bingo_idx = check_bingo(table)
    if x in x_bingo_idx:
        return True
    elif y in y_bingo_idx:
        return True
    elif DIAG_IDX in x_bingo_idx:
        if ((x==0 and y==0) or
            (x==1 and y==1) or
            (x==2 and y==2) or
            (x==3 and y==3) or
            (x==4 and y==4)):
            return True
    elif DIAG_IDX in y_bingo_idx:
        if ((x==0 and y==4) or
            (x==1 and y==3) or
            (x==2 and y==2) or
            (x==3 and y==1) or
            (x==4 and y==0)):
            return True
    else:
        return False


def bomb_explode(table, bomb_pos):
    """
    bomb_pos : (x, y)
    """
    next_table = table.copy()
    x, y = bomb_pos
    x_bingo, y_bingo = check_bingo(table)
    if x>0:
        next_table[x-1,y] = not next_table[x-1,y]
    if x<4:
        next_table[x+1,y] = not next_table[x+1,y]
    if y>0:
        next_table[x,y-1] = not next_table[x,y-1]
    if y<4:
        next_table[x,y+1] = not next_table[x,y+1]
    next_table[x,y] = not next_table[x,y]
    # Fix bingo
    for x_idx in x_bingo:
        if x_idx!=DIAG_IDX:
            next_table[x_idx,:] = True
        else:
            next_table[0,0] = True
            next_table[1,1] = True
            next_table[2,2] = True
            next_table[3,3] = True
            next_table[4,4] = True
    for y_idx in y_bingo:
        if y_idx!=DIAG_IDX:
            next_table[:,y_idx] = True
        else:
            next_table[0,4] = True
            next_table[1,3] = True
            next_table[2,2] = True
            next_table[3,1] = True
            next_table[4,0] = True

    return next_table


def fill_table(initial_table):
    import tqdm
    state_stack = [(initial_table,0)]
    stack_tqdm = tqdm.tqdm(total=1)
    max_stack = 1
    loop_n = 0
    # Dummy
    checked = [[0]*25+[0]]
    while len(state_stack)>0:
        current_table, step = state_stack[-1]
        if step!=2:
            step_idx = step%3
        else:
            # In inanna mode, step 2 is special
            step_idx = 3
        current_index = np.concatenate((current_table.reshape(-1),[step_idx]))
        current_index = tuple(current_index)
        before_bingo = count_bingo(current_table)
        before_bingo_x, before_bingo_y = check_bingo(current_table)
                
                
        possible_choices = []
        need_to_fill = False
        for action_x in range(5):
            for action_y in range(5):
                if is_in_bingo(current_table, action_x, action_y):
                    recommandable = 0
                else:
                    recommandable = 1
                next_table = bomb_explode(current_table, (action_x,action_y))
                next_bingo = count_bingo(next_table)
                next_bingo_x, next_bingo_y = check_bingo(next_table)
                assert next_bingo>=before_bingo, 'Bingo number decreased!'
                
                new_bingo = next_bingo - before_bingo
                if step==1:
                    next_step_idx = 3
                else:
                    next_step_idx = (step+1)%3
                next_index_np = np.append(next_table.reshape(-1),next_step_idx)
                next_index = tuple(next_index_np)

                if next_bingo ==0:
                    if not np.any(np.all(checked==next_index_np,axis=1)):
                        checked.append(next_index)
                        q_filled[next_index] = False
                # Ignore step == 2 : Inanna version
                if (step%3 == 2) and new_bingo == 0 and (step>2):
                    # Game over
                    possible_choices.append([
                        action_x,
                        action_y,
                        0,
                        0,
                        0,
                        recommandable,
                    ])
                elif q_filled[next_index]:
                    if (step%3==2) and new_bingo>0 and (step>2):
                        # 무력 성공
                        possible_choices.append([
                            action_x,
                            action_y,
                            q_table[next_index][2]+1,
                            q_table[next_index][3]\
                                +bingo_point(before_bingo_x, next_bingo_x)\
                                +bingo_point(before_bingo_y, next_bingo_y)\
                                +B_N_PENALTY*(new_bingo-1),
                            q_table[next_index][4]+skull_point(next_table),
                            recommandable,
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
                            q_table[next_index][4]+skull_point(next_table),
                            recommandable,
                        ])
                else:
                    state_stack.append((next_table, step+1))
                    need_to_fill = True
        if not need_to_fill:
            # All checked
            best_choices = []
            max_weak = 0
            # over weak_limit is treated the same
            current_weak_limit = max(0,WEAK_LIMIT-(step//3))
            # if there's any recommandable, that is the primary selection
            recommandable_choices = []
            for pc in possible_choices:
                if pc[5]==1:
                    recommandable_choices.append(pc)
            if len(recommandable_choices)>0:
                possible_choices = recommandable_choices
            for pc in possible_choices:
                if max_weak<pc[2] and max_weak<current_weak_limit:
                    best_choices = []
                    best_choices.append(pc)
                    max_weak = min(pc[2],current_weak_limit)
                elif (max_weak==pc[2] or 
                    (max_weak==current_weak_limit and pc[2]>=max_weak)):
                    best_choices.append(pc)
            min_bingo_p = best_choices[0][3]
            best_choices_2 = []
            for bc in best_choices:
                if min_bingo_p>bc[3]:
                    best_choices_2=[]
                    best_choices_2.append(bc)
                    min_bingo_p = bc[3]
                elif min_bingo_p == bc[3]:
                    best_choices_2.append(bc)
            min_skull_p = best_choices_2[0][4]
            best_choices_3 = []
            for bc2 in best_choices_2:
                if min_skull_p>bc2[4]:
                    best_choices_3=[]
                    best_choices_3.append(bc2)
                    min_skull_p = bc2[4]
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
        stack_tqdm.set_postfix(checked = len(checked))
        stack_tqdm.update(n=0)
    stack_tqdm.close()

if __name__ == '__main__':
    from time import time
    import datetime
    st = time()
    for p1 in range(24):
        for p2 in range(p1+1,25):
            x1 = p1//5
            y1 = p1%5
            x2 = p2//5
            y2 = p2%5
            print([x1,y1,x2,y2])
            initial_table = np.zeros((5,5),dtype=np.bool)
            initial_table[x1,y1] = True
            initial_table[x2,y2] = True
            fill_table(initial_table)
            print(str(datetime.timedelta(seconds=time()-st)))
    np.savez_compressed('bingo_tables/bingo_table_inanna_neverbingo.npz',
                        table=q_table, filled=q_filled)
    # test=bomb_explode(initial_table, [3,1])
    # test=bomb_explode(test,[3,4])
    # print(test)
    # print('skull p :',skull_point(test))
    # x_bin, y_bin = check_bingo(test)
    # test=bomb_explode(test,[0,0])
    # print(test)
    # print('skull p :',skull_point(test))
    # x_bin2, y_bin2 = check_bingo(test)

    # print('x bingo p:',bingo_point(x_bin, x_bin2))
    # print('y bingo p:',bingo_point(y_bin, y_bin2))

