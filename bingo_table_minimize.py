import numpy as np
loaded = np.load('bingo_tables/bingo_table_super.npz')
filleds=[]
tables = []
for k, v in loaded.items():
    print(k)
    if k=='inanna_no_filled':
        print('no filled')
        tmp_filled = np.zeros([2]*25+[6],dtype=np.bool)
        tmp_filled[...,:3] = v
        filleds.append(tmp_filled)
    elif k=='inanna_no_table':
        print('no table')
        tmp_table = np.zeros([2]*25+[6,3],dtype=np.uint8)
        tmp_table[...,:3,:] = v[...,:3].astype(np.uint8)
        tables.append(tmp_table)
    elif len(v.shape)==26: #filled
        filleds.append(v)
    elif len(v.shape)==27:
        tables.append(v[...,:3].astype(np.uint8))
    else:
        raise ValueError
minimized_filled = np.stack(filleds,axis=0)
minimized_table = np.stack(tables,axis=0)
np.savez_compressed(
    'bingo_tables/bingo_table_minimized.npz',
    table=minimized_table,
    filled=minimized_filled
)