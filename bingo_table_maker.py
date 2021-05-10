import numpy as np

q_table = np.zeros([2]*25+[3,25,3])
q_filled = np.zeros([2]*25+[3,25], dtype=np.bool)
MAX_STEPS = 18
EMPTY_BOARD = np.zeros((5,5), dtype=np.bool)

def fill_table(initial_table):
    pass
