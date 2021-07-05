import numpy as np
from pathlib import Path
from tqdm import trange

def bingo_name(i):
    return f'bingo_table_inanna_neverbingo_multi_{i}.npz'

BINGO_DIR = Path('bingo_tables/nth')
inanna_no = np.load('bingo_tables/dist/bingo_table_nonanna_nobingo.npz')
inanna_no_table = inanna_no['table']
inanna_no_filled = inanna_no['filled']

inanna_0 = np.load('bingo_tables/dist/bingo_table_inanna_neverbingo_fixed2.npz')
inanna_0_table = inanna_0['table']
inanna_0_filled = inanna_0['filled']

save_files = {}
save_files['inanna_no_filled'] = inanna_no_filled
save_files['inanna_no_table'] = inanna_no_table
save_files['inanna_0_filled'] = inanna_0_filled
save_files['inanna_0_table'] = inanna_0_table

for i in trange(5):
    for j in trange(6):
        if j==0:
            loaded = np.load(str(BINGO_DIR/bingo_name(i*6+j)))
            merged_table = loaded['table']
            merged_filled = loaded['filled']
        else:
            loaded = np.load(str(BINGO_DIR/bingo_name(i*6+j)))
            merged_table = np.where(
                np.logical_not(merged_filled)[...,np.newaxis],
                merged_table,
                loaded['table']
            )
            merged_filled = np.logical_or(merged_filled, loaded['filled'])
    save_files[f'inanna_{i+1}_filled'] = merged_filled
    save_files[f'inanna_{i+1}_table'] = merged_table
np.savez_compressed('bingo_table_super.npz',**save_files)