import numpy as np
loaded = np.load('bingo_tables/bingo_table_super.npz')
minimized = {}
for k, v in loaded.items():
    print(k)
    if len(v.shape)==26: #filled
        minimized[k] = v
    elif len(v.shape)==27:
        minimized[k] = v[...,:3].astype(np.uint8)
    else:
        raise ValueError
np.savez_compressed('bingo_tables/bingo_table_minimized.npz',**minimized)