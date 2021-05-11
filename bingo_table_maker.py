import numpy as np

q_table = np.zeros([2]*25+[3,25,6])
q_filled = np.zeros([2]*25+[3,25], dtype=np.bool)
MAX_STEPS = 18
EMPTY_BOARD = np.zeros((5,5), dtype=np.bool)

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
    while len(state_stack)>0:
        current_table, step = state_stack[-1]
        possible_choices = []
        need_to_fill = False
        for action_x in range(5):
            for action_y in range(5):
                next_table = bomb_explode(current_table, (action_x,action_y))
                before_bingo = count_bingo(current_table)
                next_bingo = count_bingo(next_table)
                